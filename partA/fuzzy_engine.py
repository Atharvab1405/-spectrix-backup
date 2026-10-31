"""Configurable Mamdani fuzzy engine for channel suitability decisions.

SNR is consumed in dB and interference power in dBm, matching
``SensingReading`` and the shared simulator calculations. The initial domains,
membership boundaries, rule table, and output score are configurable design
assumptions; they are not presented as universal radio thresholds or measured
research values.

In the current simulator, ``interference_dbm is None`` means no active modeled
primary-signal powers were summed. By default this explicit no-modeled-
interference case maps to -120 dBm. A missing measurement is different and must
be signaled with ``interference_available=False``; it raises an error. Numeric
0 dBm remains a numeric value and is clipped to the configured upper domain,
not treated as absence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import skfuzzy as fuzz


Category = Literal["low", "medium", "high"]
CATEGORIES: tuple[Category, ...] = ("low", "medium", "high")


@dataclass(frozen=True, slots=True)
class MembershipSet:
    """Membership function vertices for the low/medium/high categories."""

    low: tuple[float, ...]
    medium: tuple[float, ...]
    high: tuple[float, ...]

    def for_category(self, category: Category) -> tuple[float, ...]:
        return getattr(self, category)


@dataclass(frozen=True, slots=True)
class FuzzyRule:
    """One complete three-antecedent Mamdani rule."""

    occupancy: Category
    snr: Category
    interference: Category
    suitability: Category

    @property
    def rule_id(self) -> str:
        return (
            f"occupancy_{self.occupancy}_snr_{self.snr}_"
            f"interference_{self.interference}_to_{self.suitability}"
        )


def _make_default_rules() -> tuple[FuzzyRule, ...]:
    """Return the explicit default 27-combination rule table."""

    output_by_input: dict[tuple[str, str, str], Category] = {
        # Low occupancy: quality and interference determine channel suitability.
        ("low", "low", "low"): "medium",
        ("low", "low", "medium"): "low",
        ("low", "low", "high"): "low",
        ("low", "medium", "low"): "high",
        ("low", "medium", "medium"): "medium",
        ("low", "medium", "high"): "low",
        ("low", "high", "low"): "high",
        ("low", "high", "medium"): "high",
        ("low", "high", "high"): "medium",
        # Medium occupancy limits suitability even with favorable radio quality.
        ("medium", "low", "low"): "low",
        ("medium", "low", "medium"): "low",
        ("medium", "low", "high"): "low",
        ("medium", "medium", "low"): "medium",
        ("medium", "medium", "medium"): "low",
        ("medium", "medium", "high"): "low",
        ("medium", "high", "low"): "high",
        ("medium", "high", "medium"): "medium",
        ("medium", "high", "high"): "low",
        # High occupancy always maps to low suitability (primary-user protection).
        ("high", "low", "low"): "low",
        ("high", "low", "medium"): "low",
        ("high", "low", "high"): "low",
        ("high", "medium", "low"): "low",
        ("high", "medium", "medium"): "low",
        ("high", "medium", "high"): "low",
        ("high", "high", "low"): "low",
        ("high", "high", "medium"): "low",
        ("high", "high", "high"): "low",
    }
    return tuple(
        FuzzyRule(
            occupancy=occupancy,  # type: ignore[arg-type]
            snr=snr,  # type: ignore[arg-type]
            interference=interference,  # type: ignore[arg-type]
            suitability=suitability,
        )
        for (occupancy, snr, interference), suitability in output_by_input.items()
    )


DEFAULT_RULE_TABLE = _make_default_rules()


@dataclass(frozen=True, slots=True)
class FuzzyEngineConfig:
    """Configurable ranges, membership functions, rules, and None semantics."""

    snr_domain: tuple[float, float] = (-10.0, 50.0)
    snr_memberships: MembershipSet = MembershipSet(
        low=(-10.0, -10.0, 5.0, 15.0),
        medium=(5.0, 25.0, 40.0),
        high=(30.0, 40.0, 50.0, 50.0),
    )
    occupancy_domain: tuple[float, float] = (0.0, 1.0)
    occupancy_memberships: MembershipSet = MembershipSet(
        low=(0.0, 0.0, 0.20, 0.45),
        medium=(0.25, 0.50, 0.75),
        high=(0.55, 0.80, 1.0, 1.0),
    )
    interference_domain: tuple[float, float] = (-120.0, -40.0)
    interference_memberships: MembershipSet = MembershipSet(
        low=(-120.0, -120.0, -105.0, -85.0),
        medium=(-100.0, -80.0, -60.0),
        high=(-75.0, -55.0, -40.0, -40.0),
    )
    suitability_domain: tuple[float, float] = (0.0, 100.0)
    suitability_memberships: MembershipSet = MembershipSet(
        low=(0.0, 0.0, 20.0, 40.0),
        medium=(25.0, 50.0, 75.0),
        high=(60.0, 80.0, 100.0, 100.0),
    )
    none_interference_dbm: float = -120.0
    rules: tuple[FuzzyRule, ...] = DEFAULT_RULE_TABLE

    def __post_init__(self) -> None:
        for name, domain, functions in (
            ("SNR", self.snr_domain, self.snr_memberships),
            ("occupancy", self.occupancy_domain, self.occupancy_memberships),
            ("interference", self.interference_domain, self.interference_memberships),
            ("suitability", self.suitability_domain, self.suitability_memberships),
        ):
            _validate_memberships(name, domain, functions)
        if not math.isfinite(self.none_interference_dbm):
            raise ValueError("none_interference_dbm must be finite")
        if not self.interference_domain[0] <= self.none_interference_dbm <= self.interference_domain[1]:
            raise ValueError("none_interference_dbm must be inside the interference domain")

        expected = {
            (occupancy, snr, interference)
            for occupancy in CATEGORIES
            for snr in CATEGORIES
            for interference in CATEGORIES
        }
        actual = {
            (rule.occupancy, rule.snr, rule.interference)
            for rule in self.rules
        }
        if len(self.rules) != 27 or actual != expected:
            raise ValueError("rules must define each of the 27 input combinations exactly once")
        if any(rule.suitability not in CATEGORIES for rule in self.rules):
            raise ValueError("each rule suitability must be low, medium, or high")


@dataclass(frozen=True, slots=True)
class FiredRule:
    """Explanation for one rule with non-zero activation."""

    rule_id: str
    suitability: Category
    activation: float


@dataclass(frozen=True, slots=True)
class FuzzyScoreResult:
    """Channel suitability plus auditable input and rule explanations."""

    suitability_score: float
    snr_db: float
    occupancy_probability: float
    interference_dbm: float
    interference_interpretation: str
    clipped_inputs: tuple[str, ...]
    fired_rules: tuple[FiredRule, ...]


def _validate_memberships(
    label: str,
    domain: tuple[float, float],
    memberships: MembershipSet,
) -> None:
    if len(domain) != 2 or not all(math.isfinite(value) for value in domain):
        raise ValueError(f"{label} domain must contain two finite values")
    low, high = domain
    if low >= high:
        raise ValueError(f"{label} domain lower bound must be below upper bound")
    for category in CATEGORIES:
        points = memberships.for_category(category)
        if len(points) not in (3, 4):
            raise ValueError(f"{label} {category} membership needs 3 or 4 points")
        if not all(math.isfinite(value) for value in points):
            raise ValueError(f"{label} {category} membership points must be finite")
        if tuple(sorted(points)) != points:
            raise ValueError(f"{label} {category} membership points must be ordered")
        if points[0] < low or points[-1] > high:
            raise ValueError(f"{label} {category} membership points must fit its domain")


def _membership_values(
    value: float,
    memberships: MembershipSet,
) -> dict[Category, float]:
    values: dict[Category, float] = {}
    for category in CATEGORIES:
        points = np.asarray(memberships.for_category(category), dtype=float)
        if len(points) == 3:
            membership = fuzz.trimf(np.asarray([value]), points)[0]
        else:
            membership = fuzz.trapmf(np.asarray([value]), points)[0]
        values[category] = float(membership)
    return values


class FuzzyChannelDecisionEngine:
    """Score channel suitability using configurable Mamdani fuzzy inference."""

    def __init__(self, config: FuzzyEngineConfig | None = None) -> None:
        self.config = config or FuzzyEngineConfig()

    @staticmethod
    def _numeric_input(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        return number

    @staticmethod
    def _clip(
        name: str,
        value: float,
        domain: tuple[float, float],
        clipped_inputs: list[str],
    ) -> float:
        clipped = min(domain[1], max(domain[0], value))
        if clipped != value:
            clipped_inputs.append(name)
        return clipped

    def score(
        self,
        *,
        snr_db: float,
        occupancy_probability: float,
        interference_dbm: float | None,
        interference_available: bool = True,
    ) -> FuzzyScoreResult:
        """Calculate a channel suitability score from one HMM/sensing input set.

        ``interference_available=False`` means the measurement is missing and
        raises an error. ``interference_dbm=None`` with an available measurement
        means the simulator modeled no primary interference and uses the
        configured no-interference floor. A numeric zero is treated as 0 dBm,
        then clipped to the upper end of the configured dBm domain.
        """

        if not isinstance(interference_available, bool):
            raise TypeError("interference_available must be a boolean")
        if not interference_available:
            raise ValueError("interference measurement is missing")

        snr = self._numeric_input("snr_db", snr_db)
        occupancy = self._numeric_input("occupancy_probability", occupancy_probability)

        if interference_dbm is None:
            interference = self.config.none_interference_dbm
            interpretation = "no_modeled_interference_mapped_to_configured_floor"
        else:
            interference = self._numeric_input("interference_dbm", interference_dbm)
            interpretation = "measured_or_simulated_interference_power_dbm"

        clipped_inputs: list[str] = []
        snr = self._clip("snr_db", snr, self.config.snr_domain, clipped_inputs)
        occupancy = self._clip(
            "occupancy_probability",
            occupancy,
            self.config.occupancy_domain,
            clipped_inputs,
        )
        interference = self._clip(
            "interference_dbm",
            interference,
            self.config.interference_domain,
            clipped_inputs,
        )

        memberships = {
            "occupancy": _membership_values(occupancy, self.config.occupancy_memberships),
            "snr": _membership_values(snr, self.config.snr_memberships),
            "interference": _membership_values(
                interference, self.config.interference_memberships
            ),
        }
        output_universe = np.linspace(
            self.config.suitability_domain[0],
            self.config.suitability_domain[1],
            1001,
        )
        aggregated_output = np.zeros_like(output_universe)
        fired_rules: list[FiredRule] = []

        for rule in self.config.rules:
            activation = min(
                memberships["occupancy"][rule.occupancy],
                memberships["snr"][rule.snr],
                memberships["interference"][rule.interference],
            )
            if activation <= 0.0:
                continue
            fired_rules.append(
                FiredRule(
                    rule_id=rule.rule_id,
                    suitability=rule.suitability,
                    activation=float(activation),
                )
            )
            output_points = np.asarray(
                self.config.suitability_memberships.for_category(rule.suitability),
                dtype=float,
            )
            if len(output_points) == 3:
                consequent = fuzz.trimf(output_universe, output_points)
            else:
                consequent = fuzz.trapmf(output_universe, output_points)
            aggregated_output = np.fmax(
                aggregated_output,
                np.fmin(activation, consequent),
            )

        if not fired_rules or not np.any(aggregated_output):
            raise RuntimeError("no fuzzy rules fired for the supplied inputs")
        score = float(fuzz.defuzz(output_universe, aggregated_output, "centroid"))
        return FuzzyScoreResult(
            suitability_score=score,
            snr_db=snr,
            occupancy_probability=occupancy,
            interference_dbm=interference,
            interference_interpretation=interpretation,
            clipped_inputs=tuple(clipped_inputs),
            fired_rules=tuple(fired_rules),
        )
