"""K-of-N cooperative fusion for existing SPECTRIX sensing observations.

The shared simulator creates each SU's probabilistic observation using the
ground-truth occupancy and its configured false-alarm/missed-detection rates.
That observation model is not a threshold-based energy detector. This module
only combines the resulting ``sensed_busy`` reports; it neither reads ground
truth nor generates sensor measurements.

By default, a channel is classified busy only when a strict majority of the
reports received for that channel say busy. An explicit ``k`` sets the busy
vote threshold for every channel. For a fixed ``k``, each channel must have at
least ``k`` reports. With the default, the majority threshold is calculated
from the number of reports actually received for that channel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from simulator.models import SensingReading


@dataclass(frozen=True, slots=True)
class ChannelFusionResult:
    """Cooperative classification and retained SU reports for one channel."""

    channel_id: int
    individual_observations: tuple[SensingReading, ...]
    busy_votes: int
    report_count: int
    required_busy_votes: int
    fused_busy: bool
    disagreement: bool


@dataclass(frozen=True, slots=True)
class CooperativeSensingResult:
    """Fusion output for one simulator time slot."""

    time_slot: int | None
    channels: tuple[ChannelFusionResult, ...]

    def for_channel(self, channel_id: int) -> ChannelFusionResult:
        """Return a channel result or raise ``KeyError`` if it was not reported."""

        for result in self.channels:
            if result.channel_id == channel_id:
                return result
        raise KeyError(f"no sensing reports for channel {channel_id}")


class CooperativeSensingFusion:
    """Combine per-SU reports using configurable K-of-N fusion.

    Args:
        k: Required busy votes. ``None`` selects strict majority separately
            for each channel, using ``floor(report_count / 2) + 1``.

    The input should contain the readings from one snapshot. Channel and user
    IDs are zero-based integers in the shared simulator, and any positive
    number of channels is supported. If report counts differ by channel,
    default-majority thresholds are calculated independently per channel.
    """

    def __init__(self, k: int | None = None) -> None:
        if k is not None:
            if isinstance(k, bool) or not isinstance(k, int):
                raise TypeError("k must be a positive integer or None")
            if k < 1:
                raise ValueError("k must be at least 1")
        self.k = k

    def fuse(
        self,
        readings: Sequence[SensingReading],
        *,
        time_slot: int | None = None,
    ) -> CooperativeSensingResult:
        """Fuse the individual busy/idle reports supplied for one time slot.

        The function reads only ``secondary_user_id``, ``channel_id``, and
        ``sensed_busy`` from each observation. It rejects duplicate reports
        from the same SU for a channel, invalid identifiers, and a configured
        ``k`` larger than a channel's available report count.
        """

        if time_slot is not None:
            if isinstance(time_slot, bool) or not isinstance(time_slot, int):
                raise TypeError("time_slot must be a non-negative integer or None")
            if time_slot < 0:
                raise ValueError("time_slot must not be negative")

        reports = tuple(readings)
        if not reports:
            raise ValueError("at least one sensing reading is required")

        by_channel: dict[int, list[SensingReading]] = {}
        seen_pairs: set[tuple[int, int]] = set()
        for reading in reports:
            if not isinstance(reading, SensingReading):
                raise TypeError("readings must contain SensingReading objects")
            channel_id = reading.channel_id
            user_id = reading.secondary_user_id
            if isinstance(channel_id, bool) or not isinstance(channel_id, int):
                raise TypeError("channel_id values must be integers")
            if channel_id < 0:
                raise ValueError("channel_id values must not be negative")
            if isinstance(user_id, bool) or not isinstance(user_id, int):
                raise TypeError("secondary_user_id values must be integers")
            if user_id < 0:
                raise ValueError("secondary_user_id values must not be negative")
            if not isinstance(reading.sensed_busy, bool):
                raise TypeError("sensed_busy values must be booleans")

            user_channel = (user_id, channel_id)
            if user_channel in seen_pairs:
                raise ValueError(
                    f"duplicate report from SU{user_id} for channel {channel_id}"
                )
            seen_pairs.add(user_channel)
            by_channel.setdefault(channel_id, []).append(reading)

        channel_results: list[ChannelFusionResult] = []
        for channel_id in sorted(by_channel):
            observations = tuple(
                sorted(
                    by_channel[channel_id],
                    key=lambda reading: reading.secondary_user_id,
                )
            )
            report_count = len(observations)
            required_busy_votes = (
                report_count // 2 + 1 if self.k is None else self.k
            )
            if required_busy_votes > report_count:
                raise ValueError(
                    f"k={required_busy_votes} exceeds the {report_count} reports "
                    f"available for channel {channel_id}"
                )

            busy_votes = sum(reading.sensed_busy for reading in observations)
            channel_results.append(
                ChannelFusionResult(
                    channel_id=channel_id,
                    individual_observations=observations,
                    busy_votes=busy_votes,
                    report_count=report_count,
                    required_busy_votes=required_busy_votes,
                    fused_busy=busy_votes >= required_busy_votes,
                    disagreement=0 < busy_votes < report_count,
                )
            )

        return CooperativeSensingResult(
            time_slot=time_slot,
            channels=tuple(channel_results),
        )