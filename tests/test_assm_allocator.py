"""Tests for the Phase 7 ASSM gating and channel allocation."""

from __future__ import annotations

import unittest

from experiments.policy_runner import DecisionSnapshot
from partA.assm_allocator import AssmAllocationConfig, AssmAllocator
from partA.cooperative_sensing import CooperativeSensingFusion
from simulator.models import SensingReading


def make_snapshot(busy_reports: dict[int, tuple[int, ...]] | None = None) -> DecisionSnapshot:
    busy_reports = busy_reports or {}
    readings = tuple(
        SensingReading(
            secondary_user_id=user_id,
            channel_id=channel_id,
            sensed_busy=user_id in busy_reports.get(channel_id, ()),
            rssi_dbm=-95.0,
            snr_db=28.0,
            sinr_db=25.0,
            interference_dbm=None,
        )
        for user_id in range(3)
        for channel_id in range(4)
    )
    return DecisionSnapshot(
        time_slot=0,
        packet_arrivals=(0, 0, 0),
        sensing_readings=readings,
        queue_lengths=(1, 1, 1),
    )


class AssmAllocatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fusion = CooperativeSensingFusion()
        self.allocator = AssmAllocator(
            config=AssmAllocationConfig(
                reject_busy_probability=0.50,
                defer_busy_probability=0.45,
                minimum_suitability_score=0.0,
            )
        )

    def test_cooperative_busy_and_hmm_thresholds_gate_channels(self) -> None:
        snapshot = make_snapshot({0: (0, 1)})
        fused = self.fusion.fuse(snapshot.sensing_readings)
        probabilities = {0: 0.10, 1: 0.44, 2: 0.50, 3: 0.47}

        result = self.allocator.allocate(snapshot, fused, probabilities)

        decisions = {
            (assessment.secondary_user_id, assessment.channel_id): assessment.decision
            for assessment in result.assessments
        }
        self.assertEqual(decisions[(0, 0)], "COOPERATIVE_BUSY_REJECT")
        self.assertEqual(decisions[(0, 1)], "SELECTED")
        self.assertEqual(decisions[(0, 2)], "HMM_BUSY_PROBABILITY_REJECT")
        self.assertEqual(decisions[(0, 3)], "HMM_UNCERTAINTY_DEFER")

    def test_selected_channels_are_unique_and_higher_suitability_wins_ties(self) -> None:
        snapshot = make_snapshot()
        fused = self.fusion.fuse(snapshot.sensing_readings)
        result = self.allocator.allocate(
            snapshot,
            fused,
            {channel_id: 0.10 for channel_id in range(4)},
        )

        selected = [channel for channel in result.selected_channels if channel is not None]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(result.selected_channels, (0, 1, 2))

    def test_decision_snapshot_does_not_expose_primary_ground_truth(self) -> None:
        snapshot = make_snapshot()

        self.assertFalse(hasattr(snapshot, "primary_occupancy"))
        self.assertTrue(hasattr(snapshot, "sensing_readings"))
        self.assertTrue(hasattr(snapshot, "queue_lengths"))

    def test_invalid_probability_or_thresholds_are_rejected(self) -> None:
        snapshot = make_snapshot()
        fused = self.fusion.fuse(snapshot.sensing_readings)
        with self.assertRaisesRegex(ValueError, "probability"):
            self.allocator.allocate(
                snapshot,
                fused,
                {0: 0.1, 1: 0.1, 2: 1.1, 3: 0.1},
            )
        with self.assertRaisesRegex(ValueError, "thresholds"):
            AssmAllocationConfig(
                reject_busy_probability=0.4,
                defer_busy_probability=0.5,
            )


if __name__ == "__main__":
    unittest.main()
