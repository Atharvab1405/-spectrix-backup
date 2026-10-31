"""Create seeded, simulated sensing sequences aligned to database records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from partA.cooperative_sensing import CooperativeSensingFusion
from simulator import ScenarioConfig, WirelessSpectrumSimulator
from simulator.occupancy_database import (
    CHANNEL_IDS,
    EXPECTED_TIMESTAMPS,
    STEPS_PER_RECORD,
    OccupancyDatabaseReplay,
)


@dataclass(frozen=True, slots=True)
class ObservationDataset:
    """One sensing-vote sequence and separate truth sequence per channel."""

    timestamps: tuple[datetime, ...]
    observations_by_channel: dict[str, tuple[int, ...]]
    ground_truth_by_channel: dict[str, tuple[bool, ...]]
    seed: int


def normalize_busy_vote_symbol(busy_votes: int, report_count: int) -> int:
    """Map a K-of-N vote fraction to the HMM's stable 0–3 alphabet.

    With the existing three-SU configuration the mapping is exactly the
    original raw vote count. Larger scalability configurations retain the
    fractional evidence while keeping the trained categorical HMM's four
    observation symbols unchanged.
    """

    if isinstance(busy_votes, bool) or not isinstance(busy_votes, int):
        raise TypeError("busy_votes must be an integer")
    if isinstance(report_count, bool) or not isinstance(report_count, int):
        raise TypeError("report_count must be an integer")
    if report_count < 1 or not 0 <= busy_votes <= report_count:
        raise ValueError("vote count must be between zero and a positive report count")
    return (6 * busy_votes + report_count) // (2 * report_count)


def build_observation_datasets_for_user_counts(
    database_path: str | Path,
    *,
    user_counts: Sequence[int],
    seed: int = 7,
    scenario_name: str = "medium",
    start_record: int = 0,
    record_count: int | None = None,
    expected_timestamps: int = EXPECTED_TIMESTAMPS,
) -> dict[int, ObservationDataset]:
    """Build user-count-specific sensing datasets in one database pass.

    Each requested coalition gets its own simulator configured with that exact
    number of users. This preserves the existing shared RNG draw sequence for
    every user-count variant. The occupancy CSV is opened and validated once,
    and each database record is shared as ground truth for sensing generation.
    Individual SU observations and cooperative votes are never reused across
    different user counts.
    """

    counts = tuple(sorted(set(user_counts)))
    if not counts:
        raise ValueError("user_counts must contain at least one positive count")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in counts
    ):
        raise ValueError("user_counts must contain positive integers")

    source = OccupancyDatabaseReplay(
        database_path,
        expected_timestamps=expected_timestamps,
        start_record=start_record,
        record_count=record_count,
    )
    simulators = {
        count: WirelessSpectrumSimulator(
            scenario=replace(
                ScenarioConfig.preset(scenario_name),
                num_channels=len(CHANNEL_IDS),
                num_secondary_users=count,
            ),
            seed=seed,
        )
        for count in counts
    }
    fusions = {count: CooperativeSensingFusion() for count in counts}
    observations = {
        count: {channel_id: [] for channel_id in CHANNEL_IDS}
        for count in counts
    }
    ground_truth = {
        count: {channel_id: [] for channel_id in CHANNEL_IDS}
        for count in counts
    }
    timestamps: list[datetime] = []

    try:
        for local_record_index in range(source.record_count):
            record = source.record_for_slot(local_record_index * STEPS_PER_RECORD)
            timestamps.append(record.timestamp)
            for count in counts:
                readings = simulators[count].sense_occupancy(record.channel_states)
                fused = fusions[count].fuse(
                    readings,
                    time_slot=record.record_index,
                )
                for channel_index, channel_id in enumerate(CHANNEL_IDS):
                    result = fused.for_channel(channel_index)
                    observations[count][channel_id].append(
                        normalize_busy_vote_symbol(
                            result.busy_votes,
                            result.report_count,
                        )
                    )
                    ground_truth[count][channel_id].append(
                        record.channel_states[channel_index]
                    )
    finally:
        source.close()

    result_datasets: dict[int, ObservationDataset] = {}
    for count in counts:
        result_datasets[count] = ObservationDataset(
            timestamps=tuple(timestamps),
            observations_by_channel={
                channel_id: tuple(values)
                for channel_id, values in observations[count].items()
            },
            ground_truth_by_channel={
                channel_id: tuple(values)
                for channel_id, values in ground_truth[count].items()
            },
            seed=seed,
        )
    return result_datasets


def build_observation_dataset(
    database_path: str | Path,
    *,
    seed: int = 7,
    scenario_name: str = "medium",
    start_record: int = 0,
    record_count: int | None = None,
    num_secondary_users: int = 3,
    expected_timestamps: int = EXPECTED_TIMESTAMPS,
) -> ObservationDataset:
    """Sample the Phase 4 sensing model once per 10-second database record.

    Ground-truth occupancy is passed to the simulator only to generate its
    probabilistic sensor readings. The HMM observation is the cooperative
    busy-vote fraction mapped to the existing four-symbol 0–3 alphabet. For
    three SUs this is exactly the original vote count; larger coalitions are
    normalized by report count. Ground truth is stored separately and never
    passed to HMM fitting.
    """

    source = OccupancyDatabaseReplay(
        database_path,
        expected_timestamps=expected_timestamps,
        start_record=start_record,
        record_count=record_count,
    )
    scenario = replace(
        ScenarioConfig.preset(scenario_name),
        num_channels=8,
        num_secondary_users=num_secondary_users,
    )
    simulator = WirelessSpectrumSimulator(scenario=scenario, seed=seed)
    fusion = CooperativeSensingFusion()
    observations: dict[str, list[int]] = {channel_id: [] for channel_id in CHANNEL_IDS}
    ground_truth: dict[str, list[bool]] = {channel_id: [] for channel_id in CHANNEL_IDS}
    timestamps: list[datetime] = []

    try:
        for local_record_index in range(source.record_count):
            simulator_slot = local_record_index * STEPS_PER_RECORD
            record = source.record_for_slot(simulator_slot)
            readings = simulator.sense_occupancy(record.channel_states)
            fused = fusion.fuse(readings, time_slot=record.record_index)

            timestamps.append(record.timestamp)
            for channel_index, channel_id in enumerate(CHANNEL_IDS):
                channel_result = fused.for_channel(channel_index)
                observations[channel_id].append(
                    normalize_busy_vote_symbol(
                        channel_result.busy_votes,
                        channel_result.report_count,
                    )
                )
                ground_truth[channel_id].append(record.channel_states[channel_index])
    finally:
        source.close()

    return ObservationDataset(
        timestamps=tuple(timestamps),
        observations_by_channel={
            channel_id: tuple(values) for channel_id, values in observations.items()
        },
        ground_truth_by_channel={
            channel_id: tuple(values) for channel_id, values in ground_truth.items()
        },
        seed=seed,
    )
