"""Unit tests for Phase 5 categorical HMM estimation and scoring."""

from __future__ import annotations

import unittest

from partA.hmm_estimator import ChannelOccupancyHMM, classification_metrics


class ChannelOccupancyHMMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training_votes = tuple(
            [0, 0, 1, 0, 1, 0, 3, 3, 2, 3, 3, 2] * 30
        )
        self.estimator = ChannelOccupancyHMM(seed=7).fit(self.training_votes)

    def test_model_is_two_state_and_maps_busy_by_expected_vote_count(self) -> None:
        self.assertEqual(self.estimator.model.n_components, 2)
        expected = self.estimator.expected_votes_by_state
        self.assertIsNotNone(expected)
        self.assertIn(self.estimator.busy_state, (0, 1))
        self.assertGreater(expected[self.estimator.busy_state], expected[1 - self.estimator.busy_state])

    def test_next_probabilities_use_only_the_observed_prefix(self) -> None:
        prefix = (0, 1, 3, 2)
        extended = (*prefix, 0, 0, 3)
        prefix_probabilities, _ = self.estimator.predict_next_busy_probabilities(prefix)
        extended_probabilities, _ = self.estimator.predict_next_busy_probabilities(extended)

        self.assertEqual(prefix_probabilities, extended_probabilities[: len(prefix)])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in prefix_probabilities))

    def test_same_seed_fit_is_reproducible(self) -> None:
        replay = ChannelOccupancyHMM(seed=7).fit(self.training_votes)

        self.assertEqual(self.estimator.busy_state, replay.busy_state)
        self.assertEqual(self.estimator.model.startprob_.tolist(), replay.model.startprob_.tolist())
        self.assertEqual(self.estimator.model.transmat_.tolist(), replay.model.transmat_.tolist())
        self.assertEqual(
            self.estimator.model.emissionprob_.tolist(),
            replay.model.emissionprob_.tolist(),
        )

    def test_prediction_requires_a_fitted_model(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "fit the HMM"):
            ChannelOccupancyHMM().predict_next_busy_probabilities((1,))

    def test_invalid_vote_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 3"):
            ChannelOccupancyHMM().fit((0, 4, 1))


class ClassificationMetricsTests(unittest.TestCase):
    def test_confusion_counts_and_metrics(self) -> None:
        result = classification_metrics(
            (True, True, False, False),
            (True, False, True, False),
        )

        self.assertEqual(result.true_positive, 1)
        self.assertEqual(result.false_positive, 1)
        self.assertEqual(result.true_negative, 1)
        self.assertEqual(result.false_negative, 1)
        self.assertEqual(result.accuracy, 0.5)
        self.assertEqual(result.precision, 0.5)
        self.assertEqual(result.recall, 0.5)
        self.assertEqual(result.f1, 0.5)

    def test_zero_division_metrics_are_zero(self) -> None:
        result = classification_metrics((False, False), (False, False))

        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.f1, 0.0)


if __name__ == "__main__":
    unittest.main()
