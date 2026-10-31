"""Tests for Phase 4 cooperative K-of-N sensing fusion."""

from __future__ import annotations

import unittest

from partA.cooperative_sensing import CooperativeSensingFusion
from simulator import ScenarioConfig, SensingReading, WirelessSpectrumSimulator


def make_readings(states: tuple[bool, ...], channel_id: int = 0) -> tuple[SensingReading, ...]:
    """Build controlled SU observations for one channel."""

    return tuple(
        SensingReading(
            secondary_user_id=user_id,
            channel_id=channel_id,
            sensed_busy=is_busy,
            rssi_dbm=-80.0,
            snr_db=20.0,
            sinr_db=20.0,
            interference_dbm=None,
        )
        for user_id, is_busy in enumerate(states)
    )


class CooperativeSensingFusionTests(unittest.TestCase):
    def test_all_busy_reports_fuse_to_busy(self) -> None:
        result = CooperativeSensingFusion().fuse(make_readings((True, True, True)))
        channel = result.for_channel(0)

        self.assertTrue(channel.fused_busy)
        self.assertEqual(channel.busy_votes, 3)
        self.assertEqual(channel.required_busy_votes, 2)
        self.assertFalse(channel.disagreement)

    def test_all_idle_reports_fuse_to_idle(self) -> None:
        result = CooperativeSensingFusion().fuse(make_readings((False, False, False)))
        channel = result.for_channel(0)

        self.assertFalse(channel.fused_busy)
        self.assertEqual(channel.busy_votes, 0)
        self.assertEqual(channel.required_busy_votes, 2)
        self.assertFalse(channel.disagreement)

    def test_conflicting_reports_use_strict_majority(self) -> None:
        result = CooperativeSensingFusion().fuse(make_readings((True, False, False)))
        channel = result.for_channel(0)

        self.assertTrue(channel.disagreement)
        self.assertEqual(channel.busy_votes, 1)
        self.assertEqual(channel.report_count, 3)
        self.assertFalse(channel.fused_busy)

    def test_even_split_is_idle_under_strict_majority(self) -> None:
        result = CooperativeSensingFusion().fuse(make_readings((True, True, False, False)))
        channel = result.for_channel(0)

        self.assertTrue(channel.disagreement)
        self.assertEqual(channel.required_busy_votes, 3)
        self.assertFalse(channel.fused_busy)

    def test_configured_k_changes_the_fusion_threshold(self) -> None:
        readings = make_readings((True, True, False))
        channel = CooperativeSensingFusion(k=1).fuse(readings).for_channel(0)

        self.assertEqual(channel.required_busy_votes, 1)
        self.assertTrue(channel.fused_busy)

    def test_fusion_supports_four_and_eight_channels(self) -> None:
        for channel_count in (4, 8):
            readings = tuple(
                SensingReading(
                    secondary_user_id=user_id,
                    channel_id=channel_id,
                    sensed_busy=(user_id < 2),
                    rssi_dbm=-80.0,
                    snr_db=20.0,
                    sinr_db=20.0,
                    interference_dbm=None,
                )
                for channel_id in range(channel_count)
                for user_id in range(3)
            )
            result = CooperativeSensingFusion().fuse(readings)

            self.assertEqual(len(result.channels), channel_count)
            self.assertTrue(all(channel.fused_busy for channel in result.channels))

    def test_fusion_is_reproducible_for_same_seed_observations(self) -> None:
        scenario = ScenarioConfig.preset("medium")
        first_snapshot = WirelessSpectrumSimulator(scenario, seed=7).reset()
        replay_snapshot = WirelessSpectrumSimulator(scenario, seed=7).reset()
        fusion = CooperativeSensingFusion()

        first = fusion.fuse(
            first_snapshot.sensing_readings,
            time_slot=first_snapshot.time_slot,
        )
        replay = fusion.fuse(
            replay_snapshot.sensing_readings,
            time_slot=replay_snapshot.time_slot,
        )

        self.assertEqual(first, replay)

    def test_duplicate_user_channel_report_is_rejected(self) -> None:
        reading = make_readings((True,))[0]

        with self.assertRaisesRegex(ValueError, "duplicate report"):
            CooperativeSensingFusion().fuse((reading, reading))

    def test_k_larger_than_available_reports_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            CooperativeSensingFusion(k=2).fuse(make_readings((True,)))


if __name__ == "__main__":
    unittest.main()