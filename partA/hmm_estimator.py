"""Two-state categorical HMM estimation from cooperative SU vote counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from hmmlearn.hmm import CategoricalHMM


VOTE_SYMBOLS = 4
HIDDEN_STATES = 2


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Binary classification counts and derived metrics."""

    samples: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    accuracy: float
    precision: float
    recall: float
    f1: float


class ChannelOccupancyHMM:
    """Fit one two-state categorical HMM to one channel's vote-count series.

    Initial states are seeded from observations only (vote counts 0–1 as the
    lower-vote state and 2–3 as the higher-vote state), with additive smoothing.
    The final BUSY label is assigned to the hidden state with the higher
    expected busy-vote count after fitting. No ground-truth values enter fit or
    prediction methods.
    """

    def __init__(
        self,
        *,
        seed: int = 7,
        n_iter: int = 100,
        tolerance: float = 1e-4,
        smoothing: float = 1.0,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if smoothing <= 0.0:
            raise ValueError("smoothing must be positive")
        self.seed = seed
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.smoothing = smoothing
        self.model: CategoricalHMM | None = None
        self.busy_state: int | None = None
        self.expected_votes_by_state: tuple[float, float] | None = None

    @staticmethod
    def _validate_observations(observations: Sequence[int]) -> np.ndarray:
        values = np.asarray(observations)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("observations must be a non-empty one-dimensional sequence")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("vote-count observations must be integers")
        if np.any(values < 0) or np.any(values >= VOTE_SYMBOLS):
            raise ValueError("vote-count observations must be between 0 and 3")
        return values.astype(np.int64, copy=False)

    def _initialize_from_observations(self, observations: np.ndarray) -> tuple[np.ndarray, ...]:
        """Create deterministic, smoothed starting parameters without truth."""

        pseudo_states = (observations >= 2).astype(np.int64)
        counts = np.full(HIDDEN_STATES, self.smoothing, dtype=float)
        counts[pseudo_states[0]] += 1.0
        startprob = counts / counts.sum()

        transition_counts = np.full(
            (HIDDEN_STATES, HIDDEN_STATES), self.smoothing, dtype=float
        )
        for previous, current in zip(pseudo_states[:-1], pseudo_states[1:]):
            transition_counts[previous, current] += 1.0
        transmat = transition_counts / transition_counts.sum(axis=1, keepdims=True)

        emission_counts = np.full(
            (HIDDEN_STATES, VOTE_SYMBOLS), self.smoothing, dtype=float
        )
        for state, symbol in zip(pseudo_states, observations):
            emission_counts[state, symbol] += 1.0
        emissionprob = emission_counts / emission_counts.sum(axis=1, keepdims=True)
        return startprob, transmat, emissionprob

    def fit(self, observations: Sequence[int]) -> "ChannelOccupancyHMM":
        """Train from one chronological sequence of cooperative vote counts."""

        values = self._validate_observations(observations)
        startprob, transmat, emissionprob = self._initialize_from_observations(values)
        model = CategoricalHMM(
            n_components=HIDDEN_STATES,
            n_features=VOTE_SYMBOLS,
            random_state=self.seed,
            n_iter=self.n_iter,
            tol=self.tolerance,
            init_params="",
            params="ste",
        )
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.emissionprob_ = emissionprob
        model.fit(values.reshape(-1, 1), lengths=[len(values)])

        expected_votes = model.emissionprob_ @ np.arange(VOTE_SYMBOLS, dtype=float)
        busy_state = int(np.argmax(expected_votes))
        if np.isclose(expected_votes[0], expected_votes[1], rtol=0.0, atol=1e-10):
            raise ValueError(
                "the fitted hidden states have equal expected vote counts; "
                "BUSY/IDLE state mapping is ambiguous"
            )

        self.model = model
        self.busy_state = busy_state
        self.expected_votes_by_state = (
            float(expected_votes[0]),
            float(expected_votes[1]),
        )
        return self

    def predict_next_busy_probabilities(
        self,
        observations: Sequence[int],
        *,
        previous_posterior: Sequence[float] | None = None,
    ) -> tuple[tuple[float, ...], tuple[float, float]]:
        """Predict the next hidden state using observations through each row.

        ``previous_posterior`` is the filtered hidden-state distribution after
        the record immediately before this sequence. For each observation, the
        method filters the state using that observation, then transitions one
        step forward to obtain the next-record BUSY probability.
        """

        if self.model is None or self.busy_state is None:
            raise RuntimeError("fit the HMM before making predictions")
        values = self._validate_observations(observations)
        posterior = (
            None
            if previous_posterior is None
            else np.asarray(previous_posterior, dtype=float)
        )
        if posterior is not None:
            if posterior.shape != (HIDDEN_STATES,) or np.any(posterior < 0.0):
                raise ValueError("previous_posterior must contain two non-negative values")
            total = posterior.sum()
            if not np.isfinite(total) or total <= 0.0:
                raise ValueError("previous_posterior must have a positive finite sum")
            posterior = posterior / total

        next_busy_probabilities: list[float] = []
        for symbol in values:
            if posterior is None:
                filtered = self.model.startprob_ * self.model.emissionprob_[:, symbol]
            else:
                filtered = (
                    posterior @ self.model.transmat_
                ) * self.model.emissionprob_[:, symbol]
            normalizer = filtered.sum()
            if not np.isfinite(normalizer) or normalizer <= 0.0:
                raise RuntimeError("HMM filtering produced an invalid state probability")
            posterior = filtered / normalizer
            next_state_probability = posterior @ self.model.transmat_
            next_busy_probabilities.append(float(next_state_probability[self.busy_state]))

        assert posterior is not None
        return (
            tuple(next_busy_probabilities),
            (float(posterior[0]), float(posterior[1])),
        )


def classification_metrics(
    truth: Sequence[bool], predictions: Sequence[bool]
) -> ClassificationMetrics:
    """Calculate binary metrics; undefined precision/recall/F1 become zero."""

    actual = tuple(truth)
    guessed = tuple(predictions)
    if not actual or len(actual) != len(guessed):
        raise ValueError("truth and predictions must have the same non-zero length")
    if any(not isinstance(value, bool) for value in (*actual, *guessed)):
        raise TypeError("truth and predictions must contain only booleans")

    tp = sum(a and p for a, p in zip(actual, guessed))
    fp = sum((not a) and p for a, p in zip(actual, guessed))
    tn = sum((not a) and (not p) for a, p in zip(actual, guessed))
    fn = sum(a and (not p) for a, p in zip(actual, guessed))
    accuracy = (tp + tn) / len(actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ClassificationMetrics(
        samples=len(actual),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )
