"""Demonstrate optional eight-channel occupancy database playback."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from simulator import ScenarioConfig, WirelessSpectrumSimulator
from simulator.occupancy_database import OccupancyDatabaseReplay


def _collect(seed: int, steps: int, database_path: Path) -> tuple[tuple[object, ...], ...]:
    source = OccupancyDatabaseReplay(database_path)
    scenario = replace(ScenarioConfig.preset("medium"), num_channels=8)
    simulator = WirelessSpectrumSimulator(
        scenario=scenario,
        seed=seed,
        occupancy_source=source,
    )
    try:
        snapshot = simulator.reset()
        records: list[tuple[object, ...]] = []
        for slot_index in range(steps):
            source_record = source.record_for_slot(snapshot.time_slot)
            records.append(
                (
                    snapshot.time_slot,
                    source_record.timestamp.isoformat(timespec="seconds"),
                    snapshot.primary_occupancy,
                    tuple(reading.sensed_busy for reading in snapshot.sensing_readings),
                )
            )
            if slot_index + 1 < steps:
                snapshot = simulator.step()
        return tuple(records)
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the seven-day occupancy database through the shared simulator."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("results/channel_occupancy_7days.csv"),
        help="validated occupancy CSV (default: results/channel_occupancy_7days.csv)",
    )
    parser.add_argument("--steps", type=int, default=5, help="1 ms slots to display (default: 5)")
    parser.add_argument("--seed", type=int, default=7, help="traffic/sensing seed (default: 7)")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    first_run = _collect(args.seed, args.steps, args.database)
    replay = _collect(args.seed, args.steps, args.database)
    print(
        "SPECTRIX occupancy database replay | channels=8 | "
        f"seed={args.seed} | simulator_slots={args.steps} | "
        f"steps_per_database_record={OccupancyDatabaseReplay.steps_per_record}"
    )
    for time_slot, timestamp, occupancy, sensed in first_run:
        busy_channels = [f"CH{index + 1:02d}" for index, busy in enumerate(occupancy) if busy]
        sensed_busy_count = sum(sensed)
        print(
            f"slot={time_slot:05d} database_timestamp={timestamp} "
            f"PU_busy={len(busy_channels)}/8 busy_channels={busy_channels} "
            f"individual_sensed_busy={sensed_busy_count}/24"
        )
    print(f"same-seed database replay: {'PASS' if first_run == replay else 'FAIL'}")
    if first_run != replay:
        raise SystemExit("same-seed database replay did not reproduce identical output")


if __name__ == "__main__":
    main()
