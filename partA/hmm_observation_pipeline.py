"""Create seeded, simulated sensing sequences aligned to database records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dataclasses import replace

from partA.cooperative_sensing import CooperativeSensingFusion
from simulator import ScenarioConfig, WirelessSpectrumSimulator
from simulator.occupancy_database import (
    CHANNEL_IDS,
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


def build_observation_dataset(
    database_path: str | Path,
    *,
    seed: int = 7,
    scenario_name: str = "medium",
    start_record: int = 0,
    record_count: int | None = None,
) -> ObservationDataset:
    """Sample the Phase 4 sensing model once per 10-second database record.

    Ground-truth occupancy is passed to the simulator only to generate its
    probabilistic sensor readings. The HMM observation returned here is the
    cooperative count of SU reports that sensed each channel busy (0 through
    3 with the current three-SU configuration). Ground truth is stored in a
    separate mapping for evaluation and is never passed to HMM fitting.
    """

    source = OccupancyDatabaseReplay(
        database_path,
        start_record=start_record,
        record_count=record_count,
    )
    scenario = replace(ScenarioConfig.preset(scenario_name), num_channels=8)
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
                observations[channel_id].append(channel_result.busy_votes)
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
