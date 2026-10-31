"""ASSM channel gating and allocation using Phase 4-6 outputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Protocol

from partA.cooperative_sensing import CooperativeSensingResult
from partA.fuzzy_engine import FuzzyChannelDecisionEngine, FuzzyScoreResult
from simulator.models import SensingReading


class AllocationSnapshot(Protocol):
    """Decision-only snapshot; deliberately contains no primary-user truth."""

    time_slot: int
    packet_arrivals: tuple[int, ...]
    queue_lengths: tuple[int, ...]
    sensing_readings: tuple[SensingReading, ...]

    def readings_for_user(self, secondary_user_id: int) -> tuple[SensingReading, ...]:
        """Return the current-record readings for one secondary user."""


@dataclass(frozen=True, slots=True)
class AssmAllocationConfig:
    """Configurable prototype safety and suitability thresholds.

    These are initial engineering assumptions, not experimentally optimized
    thresholds. A current HMM busy probability at or above reject_busy_probability
    rejects a channel. Values in [defer_busy_probability, reject_busy_probability)
    defer. Lower probabilities are eligible if the fuzzy score also clears
    minimum_suitability_score.
    """

    reject_busy_probability: float = 0.50
    defer_busy_probability: float = 0.45
    minimum_suitability_score: float = 50.0

    def __post_init__(self) -> None:
        values = (
            self.reject_busy_probability,
            self.defer_busy_probability,
            self.minimum_suitability_score,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("ASSM thresholds must be finite")
        if not 0.0 <= self.defer_busy_probability <= self.reject_busy_probability <= 1.0:
            raise ValueError(
                "busy thresholds must satisfy 0 <= defer <= reject <= 1"
            )
        if not 0.0 <= self.minimum_suitability_score <= 100.0:
            raise ValueError("minimum_suitability_score must be between 0 and 100")


@dataclass(frozen=True, slots=True)
class AssmChannelAssessment:
    """Auditable score and gate result for one SU/channel candidate."""

    secondary_user_id: int
    channel_id: int
    busy_votes: int
    report_count: int
    cooperative_busy: bool
    hmm_busy_probability: float
    fuzzy: FuzzyScoreResult
    decision: str


@dataclass(frozen=True, slots=True)
class AssmAllocationResult:
    """Unique per-user channel allocation plus all candidate assessments."""

    selected_channels: tuple[int | None, ...]
    assessments: tuple[AssmChannelAssessment, ...]


class AssmAllocator:
    """Allocate distinct channels after cooperative, HMM, and fuzzy checks."""

    def __init__(
        self,
        fuzzy_engine: FuzzyChannelDecisionEngine | None = None,
        config: AssmAllocationConfig | None = None,
    ) -> None:
        self.fuzzy_engine = fuzzy_engine or FuzzyChannelDecisionEngine()
        self.config = config or AssmAllocationConfig()

    def allocate(
        self,
        snapshot: AllocationSnapshot,
        cooperative: CooperativeSensingResult,
        hmm_busy_probabilities: dict[int, float],
    ) -> AssmAllocationResult:
        """Choose highest-scoring eligible channels in ascending SU ID order.

        Inputs are only the decision-only snapshot, cooperative report fusion,
        current-record filtered HMM probabilities, and fuzzy scores. Simulator
        ground truth is not part of this interface.
        """

        user_count = len(snapshot.packet_arrivals)
        if user_count < 1 or len(snapshot.queue_lengths) != user_count:
            raise ValueError("snapshot queue lengths must match its users")
        readings_by_channel = {
            result.channel_id: result for result in cooperative.channels
        }
        channel_ids = sorted(readings_by_channel)
        if set(hmm_busy_probabilities) != set(channel_ids):
            raise ValueError("HMM probabilities must cover exactly the fused channels")

        assessments: dict[tuple[int, int], AssmChannelAssessment] = {}
        for user_id in range(user_count):
            readings = snapshot.readings_for_user(user_id)
            if {reading.channel_id for reading in readings} != set(channel_ids):
                raise ValueError(
                    f"SU{user_id} readings do not cover the fused channel set"
                )
            reading_by_channel = {reading.channel_id: reading for reading in readings}

            for channel_id in channel_ids:
                fusion = readings_by_channel[channel_id]
                probability = hmm_busy_probabilities[channel_id]
                if (
                    isinstance(probability, bool)
                    or not isinstance(probability, (int, float))
                    or not math.isfinite(float(probability))
                    or not 0.0 <= probability <= 1.0
                ):
                    raise ValueError(
                        f"HMM busy probability for channel {channel_id} must be in [0, 1]"
                    )

                reading = reading_by_channel[channel_id]
                fuzzy = self.fuzzy_engine.score(
                    snr_db=reading.snr_db,
                    occupancy_probability=float(probability),
                    interference_dbm=reading.interference_dbm,
                )
                if fusion.fused_busy:
                    reason = "COOPERATIVE_BUSY_REJECT"
                elif probability >= self.config.reject_busy_probability:
                    reason = "HMM_BUSY_PROBABILITY_REJECT"
                elif probability >= self.config.defer_busy_probability:
                    reason = "HMM_UNCERTAINTY_DEFER"
                elif fuzzy.suitability_score < self.config.minimum_suitability_score:
                    reason = "FUZZY_SCORE_BELOW_MINIMUM"
                elif snapshot.queue_lengths[user_id] == 0:
                    reason = "NO_QUEUED_TRAFFIC"
                else:
                    reason = "ELIGIBLE"

                assessments[(user_id, channel_id)] = AssmChannelAssessment(
                    secondary_user_id=user_id,
                    channel_id=channel_id,
                    busy_votes=fusion.busy_votes,
                    report_count=fusion.report_count,
                    cooperative_busy=fusion.fused_busy,
                    hmm_busy_probability=float(probability),
                    fuzzy=fuzzy,
                    decision=reason,
                )

        selected: list[int | None] = [None] * user_count
        reserved: set[int] = set()
        for user_id in range(user_count):
            for channel_id in channel_ids:
                assessment = assessments[(user_id, channel_id)]
                if assessment.decision == "ELIGIBLE" and channel_id in reserved:
                    assessments[(user_id, channel_id)] = replace(
                        assessment, decision="CHANNEL_RESERVED_BY_EARLIER_USER"
                    )
            candidates = [
                assessments[(user_id, channel_id)]
                for channel_id in channel_ids
                if assessments[(user_id, channel_id)].decision == "ELIGIBLE"
                and channel_id not in reserved
            ]
            if not candidates:
                continue
            winner = max(
                candidates,
                key=lambda result: (
                    result.fuzzy.suitability_score,
                    -result.channel_id,
                ),
            )
            selected[user_id] = winner.channel_id
            reserved.add(winner.channel_id)
            assessments[(user_id, winner.channel_id)] = replace(
                winner, decision="SELECTED"
            )

            for candidate in candidates:
                if candidate.channel_id == winner.channel_id:
                    continue
                reason = (
                    "CHANNEL_RESERVED_BY_EARLIER_USER"
                    if candidate.channel_id in reserved
                    else "LOWER_SCORE_THAN_SELECTED"
                )
                assessments[(user_id, candidate.channel_id)] = replace(
                    candidate, decision=reason
                )

        return AssmAllocationResult(
            selected_channels=tuple(selected),
            assessments=tuple(
                assessments[(user_id, channel_id)]
                for user_id in range(user_count)
                for channel_id in channel_ids
            ),
        )
