"""Tests for SPECTRIX occupancy CSV validation and simulator playback."""

from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from simulator import ScenarioConfig, WirelessSpectrumSimulator
from simulator.occupancy_database import (
    CHANNEL_IDS,
    DATABASE_FIELDS,
    EXPECTED_TIMESTAMPS,
    STEPS_PER_RECORD,
    OccupancyDatabaseReplay,
)


class OccupancyDatabaseReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "small_occupancy.csv"
        self._write_fixture((
            (True, False, True, False, True, False, True, False),
            (False, True, False, True, False, True, False, True),
        ))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_fixture(
        self,
        occupancy_by_record: tuple[tuple[bool, ...], ...],
        *,
        timestamps: tuple[datetime, ...] | None = None,
    ) -> None:
        start = datetime.fromisoformat("2026-10-26T00:00:00+05:30")
        if timestamps is None:
            timestamps = tuple(
                start + timedelta(seconds=10 * index)
                for index in range(len(occupancy_by_record))
            )
        with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=DATABASE_FIELDS)
            writer.writeheader()
            for record_index, (timestamp, states) in enumerate(
                zip(timestamps, occupancy_by_record, strict=True)
            ):
                for channel_index, (channel_id, is_busy) in enumerate(
                    zip(CHANNEL_IDS, states, strict=True)
                ):
                    writer.writerow(
                        {
                            "timestamp": timestamp.isoformat(timespec="seconds"),
                            "day_number": 1,
                            "weekday": "Monday",
                            "hour": timestamp.hour,
                            "minute": timestamp.minute,
                            "daypart": "Nighttime",
                            "channel_id": channel_id,
                            "channel_category": "Test",
                            "occupancy_state": "busy" if is_busy else "idle",
                            "configured_busy_probability_pct": "50.00",
                        }
                    )

    def test_channel_mapping_and_busy_idle_states(self) -> None:
        replay = OccupancyDatabaseReplay(self.csv_path, expected_timestamps=2)
        first = replay.record_for_slot(0)

        self.assertEqual(CHANNEL_IDS, tuple(f"CH{index:02d}" for index in range(1, 9)))
        self.assertEqual(first.channel_states, (True, False, True, False, True, False, True, False))
        self.assertEqual(replay.state_for_slot(0)[0], True)   # CH01 -> index 0
        self.assertEqual(replay.state_for_slot(0)[7], False)  # CH08 -> index 7
        replay.close()

    def test_each_record_is_held_for_exactly_10000_slots(self) -> None:
        replay = OccupancyDatabaseReplay(self.csv_path, expected_timestamps=2)
        first_at_start = replay.record_for_slot(0)
        first_at_last_slot = replay.record_for_slot(STEPS_PER_RECORD - 1)
        second_at_boundary = replay.record_for_slot(STEPS_PER_RECORD)

        self.assertEqual(STEPS_PER_RECORD, 10_000)
        self.assertEqual(first_at_start, first_at_last_slot)
        self.assertEqual(first_at_last_slot.record_index, 0)
        self.assertEqual(second_at_boundary.record_index, 1)
        self.assertEqual(
            second_at_boundary.timestamp - first_at_last_slot.timestamp,
            timedelta(seconds=10),
        )
        replay.close()

    def test_end_of_dataset_does_not_wrap(self) -> None:
        replay = OccupancyDatabaseReplay(self.csv_path, expected_timestamps=2)
        last_slot = 2 * STEPS_PER_RECORD - 1

        self.assertEqual(replay.record_for_slot(last_slot).record_index, 1)
        with self.assertRaises(StopIteration):
            replay.record_for_slot(last_slot + 1)
        replay.close()

    def test_start_record_offset_preserves_timestamp_and_local_slot_boundaries(self) -> None:
        replay = OccupancyDatabaseReplay(
            self.csv_path,
            expected_timestamps=2,
            start_record=1,
        )
        first_local_slot = replay.record_for_slot(0)
        final_slot_of_offset_window = replay.record_for_slot(STEPS_PER_RECORD - 1)

        self.assertEqual(replay.record_count, 1)
        self.assertEqual(first_local_slot.record_index, 1)
        self.assertEqual(first_local_slot.timestamp.minute, 0)
        self.assertEqual(first_local_slot.timestamp.second, 10)
        self.assertEqual(first_local_slot, final_slot_of_offset_window)
        with self.assertRaises(StopIteration):
            replay.record_for_slot(STEPS_PER_RECORD)
        replay.close()

    def test_start_record_and_record_count_must_fit_database(self) -> None:
        with self.assertRaisesRegex(ValueError, "fit within"):
            OccupancyDatabaseReplay(
                self.csv_path,
                expected_timestamps=2,
                start_record=1,
                record_count=2,
            )

    def test_out_of_order_timestamps_are_rejected(self) -> None:
        start = datetime.fromisoformat("2026-10-26T00:00:00+05:30")
        self._write_fixture(
            (
                (True, False, True, False, True, False, True, False),
                (False, True, False, True, False, True, False, True),
            ),
            timestamps=(start, start + timedelta(seconds=20)),
        )

        with self.assertRaisesRegex(ValueError, "out of order"):
            OccupancyDatabaseReplay(self.csv_path, expected_timestamps=2)

    def test_simulator_replays_database_states_at_slot_boundaries(self) -> None:
        scenario = replace(ScenarioConfig.preset("medium"), num_channels=8)
        replay = OccupancyDatabaseReplay(self.csv_path, expected_timestamps=2)
        simulator = WirelessSpectrumSimulator(
            scenario=scenario,
            seed=7,
            occupancy_source=replay,
        )
        first = simulator.reset()
        self.assertEqual(first.time_slot, 0)
        self.assertEqual(first.primary_occupancy, replay.state_for_slot(0))

        for _ in range(STEPS_PER_RECORD - 1):
            snapshot = simulator.step()
        self.assertEqual(snapshot.time_slot, STEPS_PER_RECORD - 1)
        self.assertEqual(snapshot.primary_occupancy, first.primary_occupancy)

        second = simulator.step()
        self.assertEqual(second.time_slot, STEPS_PER_RECORD)
        self.assertEqual(
            second.primary_occupancy,
            (False, True, False, True, False, True, False, True),
        )
        replay.close()

    def test_no_database_keeps_seeded_default_simulator_reproducible(self) -> None:
        scenario = ScenarioConfig.preset("medium")
        first_simulator = WirelessSpectrumSimulator(scenario, seed=7)
        replay_simulator = WirelessSpectrumSimulator(scenario, seed=7)
        first = first_simulator.reset()
        replay = replay_simulator.reset()
        self.assertEqual(first, replay)

        first_next = first_simulator.step()
        replay_next = replay_simulator.step()
        self.assertEqual(first_next, replay_next)
        self.assertEqual(len(first_next.primary_occupancy), 4)

    def test_production_defaults_cover_seven_days(self) -> None:
        self.assertEqual(EXPECTED_TIMESTAMPS, 60_480)
        self.assertEqual(
            OccupancyDatabaseReplay.steps_per_record * EXPECTED_TIMESTAMPS,
            604_800_000,
        )


if __name__ == "__main__":
    unittest.main()