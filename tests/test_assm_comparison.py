"""Integration checks for fair paired Phase 7 execution."""

from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace as dataclass_replace
from datetime import datetime, timedelta
from pathlib import Path

from experiments.policy_runner import DecisionSnapshot, run_paired_policies
from simulator import ScenarioConfig, WirelessSpectrumSimulator
from simulator.occupancy_database import (
    CHANNEL_IDS,
    DATABASE_FIELDS,
    STEPS_PER_RECORD,
    OccupancyDatabaseReplay,
)


class IdlePolicy:
    def __init__(self, user_count: int) -> None:
        self.user_count = user_count
        self.calls: list[DecisionSnapshot] = []

    def choose_channels(
        self, snapshot: DecisionSnapshot
    ) -> tuple[int | None, ...]:
        self.calls.append(snapshot)
        if hasattr(snapshot, "primary_occupancy"):
            raise AssertionError("allocation interface leaked primary ground truth")
        return (None,) * self.user_count


class PairedPhase7RunnerTests(unittest.TestCase):
    def test_paired_policies_share_environment_and_decide_at_record_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "occupancy.csv"
            start = datetime.fromisoformat("2026-10-26T00:00:00+05:30")
            with database_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=DATABASE_FIELDS)
                writer.writeheader()
                for record_index in range(2):
                    timestamp = start + timedelta(seconds=10 * record_index)
                    for channel_id in CHANNEL_IDS:
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
                                "occupancy_state": "idle",
                                "configured_busy_probability_pct": "0.00",
                            }
                        )

            scenario = dataclass_replace(ScenarioConfig.preset("low"), num_channels=8)
            standard_source = OccupancyDatabaseReplay(
                database_path,
                expected_timestamps=2,
                record_count=2,
            )
            assm_source = OccupancyDatabaseReplay(
                database_path,
                expected_timestamps=2,
                record_count=2,
            )
            standard_policy = IdlePolicy(scenario.num_secondary_users)
            assm_policy = IdlePolicy(scenario.num_secondary_users)
            try:
                standard_metrics, assm_metrics = run_paired_policies(
                    WirelessSpectrumSimulator(
                        scenario, seed=23, occupancy_source=standard_source
                    ),
                    WirelessSpectrumSimulator(
                        scenario, seed=23, occupancy_source=assm_source
                    ),
                    standard_policy,
                    assm_policy,
                    seed=23,
                    slots=2 * STEPS_PER_RECORD,
                    decision_cadence_slots=STEPS_PER_RECORD,
                )
            finally:
                standard_source.close()
                assm_source.close()

            self.assertEqual(standard_metrics.slots, 2 * STEPS_PER_RECORD)
            self.assertEqual(assm_metrics.slots, 2 * STEPS_PER_RECORD)
            self.assertEqual(len(standard_policy.calls), 2)
            self.assertEqual(len(assm_policy.calls), 2)
            self.assertEqual(
                [view.database_record_index for view in assm_policy.calls],
                [0, 1],
            )
            self.assertEqual(
                [view.database_timestamp for view in assm_policy.calls],
                [start, start + timedelta(seconds=10)],
            )
            self.assertEqual(standard_metrics.generated_packets, assm_metrics.generated_packets)
            self.assertEqual(
                dataclass_replace(standard_metrics, method=assm_metrics.method),
                assm_metrics,
            )


if __name__ == "__main__":
    unittest.main()
