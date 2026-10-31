"""Tests for the Phase 6 configurable fuzzy channel decision engine."""

from __future__ import annotations

from dataclasses import replace
import math
import unittest

from partA.fuzzy_engine import (
    CATEGORIES,
    DEFAULT_RULE_TABLE,
    FuzzyChannelDecisionEngine,
    FuzzyEngineConfig,
    FuzzyRule,
)


class FuzzyChannelDecisionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FuzzyChannelDecisionEngine()

    def test_default_table_covers_all_27_input_combinations(self) -> None:
        combinations = {
            (rule.occupancy, rule.snr, rule.interference)
            for rule in DEFAULT_RULE_TABLE
        }
        expected = {
            (occupancy, snr, interference)
            for occupancy in CATEGORIES
            for snr in CATEGORIES
            for interference in CATEGORIES
        }

        self.assertEqual(len(DEFAULT_RULE_TABLE), 27)
        self.assertEqual(combinations, expected)

    def test_favorable_channel_gets_higher_score_than_poor_conditions(self) -> None:
        favorable = self.engine.score(
            snr_db=40.0,
            occupancy_probability=0.05,
            interference_dbm=-110.0,
        )
        low_snr = self.engine.score(
            snr_db=-5.0,
            occupancy_probability=0.05,
            interference_dbm=-110.0,
        )
        high_interference = self.engine.score(
            snr_db=40.0,
            occupancy_probability=0.05,
            interference_dbm=-45.0,
        )
        high_occupancy = self.engine.score(
            snr_db=40.0,
            occupancy_probability=0.95,
            interference_dbm=-110.0,
        )

        self.assertGreater(favorable.suitability_score, low_snr.suitability_score)
        self.assertGreater(favorable.suitability_score, high_interference.suitability_score)
        self.assertGreater(favorable.suitability_score, high_occupancy.suitability_score)
        self.assertEqual(len(high_occupancy.fired_rules), 1)
        self.assertEqual(high_occupancy.fired_rules[0].suitability, "low")

    def test_none_means_no_modeled_interference_and_zero_is_numeric(self) -> None:
        no_modeled_interference = self.engine.score(
            snr_db=25.0,
            occupancy_probability=0.2,
            interference_dbm=None,
        )
        numeric_zero = self.engine.score(
            snr_db=25.0,
            occupancy_probability=0.2,
            interference_dbm=0.0,
        )

        self.assertEqual(no_modeled_interference.interference_dbm, -120.0)
        self.assertIn("no_modeled_interference", no_modeled_interference.interference_interpretation)
        self.assertEqual(numeric_zero.interference_dbm, -40.0)
        self.assertIn("interference_dbm", numeric_zero.clipped_inputs)
        self.assertNotEqual(
            no_modeled_interference.interference_interpretation,
            numeric_zero.interference_interpretation,
        )

    def test_missing_interference_is_not_treated_as_no_interference(self) -> None:
        with self.assertRaisesRegex(ValueError, "measurement is missing"):
            self.engine.score(
                snr_db=25.0,
                occupancy_probability=0.2,
                interference_dbm=None,
                interference_available=False,
            )

    def test_out_of_range_inputs_are_clipped_and_reported(self) -> None:
        result = self.engine.score(
            snr_db=80.0,
            occupancy_probability=1.2,
            interference_dbm=-200.0,
        )

        self.assertEqual(result.snr_db, 50.0)
        self.assertEqual(result.occupancy_probability, 1.0)
        self.assertEqual(result.interference_dbm, -120.0)
        self.assertEqual(
            set(result.clipped_inputs),
            {"snr_db", "occupancy_probability", "interference_dbm"},
        )

    def test_result_contains_rule_explanations_and_is_repeatable(self) -> None:
        arguments = {
            "snr_db": 28.0,
            "occupancy_probability": 0.35,
            "interference_dbm": -85.0,
        }
        first = self.engine.score(**arguments)
        replay = self.engine.score(**arguments)

        self.assertEqual(first, replay)
        self.assertTrue(first.fired_rules)
        self.assertTrue(all(rule.rule_id and rule.activation > 0.0 for rule in first.fired_rules))
        self.assertTrue(math.isfinite(first.suitability_score))
        self.assertGreaterEqual(first.suitability_score, 0.0)
        self.assertLessEqual(first.suitability_score, 100.0)

    def test_rule_table_is_configurable(self) -> None:
        all_high_rules = tuple(
            FuzzyRule(
                occupancy=occupancy,
                snr=snr,
                interference=interference,
                suitability="high",
            )
            for occupancy in CATEGORIES
            for snr in CATEGORIES
            for interference in CATEGORIES
        )
        custom_engine = FuzzyChannelDecisionEngine(
            replace(FuzzyEngineConfig(), rules=all_high_rules)
        )
        inputs = {
            "snr_db": 28.0,
            "occupancy_probability": 0.35,
            "interference_dbm": -85.0,
        }

        self.assertGreater(
            custom_engine.score(**inputs).suitability_score,
            self.engine.score(**inputs).suitability_score,
        )

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            self.engine.score(
                snr_db=math.nan,
                occupancy_probability=0.2,
                interference_dbm=-90.0,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            # Values outside the probability interval are clipped; NaN is invalid.
            self.engine.score(
                snr_db=25.0,
                occupancy_probability=math.nan,
                interference_dbm=-90.0,
            )


if __name__ == "__main__":
    unittest.main()
