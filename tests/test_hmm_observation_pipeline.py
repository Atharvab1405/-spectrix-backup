from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from partA.hmm_observation_pipeline import (
    build_observation_dataset,
    build_observation_datasets_for_user_counts,
    normalize_busy_vote_symbol,
)
from simulator.occupancy_database import CHANNEL_IDS, DATABASE_FIELDS


class VoteSymbolNormalizationTests(unittest.TestCase):
    def test_three_user_votes_keep_the_existing_hmm_symbols(self) -> None:
        self.assertEqual(
            [normalize_busy_vote_symbol(votes, 3) for votes in range(4)],
            [0, 1, 2, 3],
        )

    def test_larger_coalitions_map_fractional_votes_to_four_symbols(self) -> None:
        for report_count in (5, 8, 10):
            symbols = [
                normalize_busy_vote_symbol(votes, report_count)
                for votes in range(report_count + 1)
            ]
            self.assertEqual(symbols[0], 0)
            self.assertEqual(symbols[-1], 3)
            self.assertTrue(all(0 <= symbol <= 3 for symbol in symbols))
            self.assertEqual(symbols, sorted(symbols))

    def test_vote_counts_outside_the_report_range_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_busy_vote_symbol(4, 3)
        with self.assertRaises(ValueError):
            normalize_busy_vote_symbol(0, 0)

    def test_user_count_datasets_match_independent_replays_across_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "occupancy.csv"
            start = datetime.fromisoformat("2026-10-26T00:00:00+05:30")
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=DATABASE_FIELDS)
                writer.writeheader()
                for record_index in range(6):
                    timestamp = start + timedelta(seconds=10 * record_index)
                    for channel_index, channel_id in enumerate(CHANNEL_IDS):
                        busy = (record_index + channel_index) % 3 != 0
                        writer.writerow(
                            {
                                "timestamp": timestamp.isoformat(timespec="seconds"),
                                "day_number": 1,
                                "weekday": "Monday",
                                "hour": 0,
                                "minute": 0,
                                "daypart": "Nighttime",
                                "channel_id": channel_id,
                                "channel_category": "Test",
                                "occupancy_state": "busy" if busy else "idle",
                                "configured_busy_probability_pct": "50.00",
                            }
                        )

            counts = (3, 5, 8, 10)
            combined = build_observation_datasets_for_user_counts(
                csv_path,
                user_counts=counts,
                seed=91,
                record_count=6,
                expected_timestamps=6,
            )
            for count in counts:
                separate = build_observation_dataset(
                    csv_path,
                    seed=91,
                    record_count=6,
                    expected_timestamps=6,
                    num_secondary_users=count,
                )
                self.assertEqual(combined[count], separate)


if __name__ == "__main__":
    unittest.main()
