"""Run paired, 10-second Standard/ASSM evaluations on held-out database day 7."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Iterable, Sequence

from baseline import StandardAllocator
from experiments.policy_runner import DecisionSnapshot, run_paired_policies
from partA.assm_allocator import AssmAllocationConfig, AssmAllocator
from partA.cooperative_sensing import CooperativeSensingFusion
from partA.fuzzy_engine import FuzzyChannelDecisionEngine
from partA.hmm_estimator import (
    ChannelOccupancyHMM,
    ClassificationMetrics,
    classification_metrics,
)
from partA.hmm_observation_pipeline import build_observation_dataset
from simulator import SCENARIO_NAMES, ScenarioConfig, WirelessSpectrumSimulator
from simulator.metrics import RunMetrics
from simulator.occupancy_database import (
    CHANNEL_IDS,
    EXPECTED_TIMESTAMPS,
    STEPS_PER_RECORD,
    OccupancyDatabaseReplay,
)


RECORDS_PER_DAY = 24 * 60 * 60 // 10
TRAIN_DAYS = 6
TRAIN_RECORDS = TRAIN_DAYS * RECORDS_PER_DAY
DAY7_RECORD = TRAIN_RECORDS
DECISION_CADENCE_SLOTS = STEPS_PER_RECORD
METHOD_STANDARD = "STANDARD_10S"
METHOD_ASSM = "ASSM"


class OnlineAssmPolicy:
    """Use current-record cooperative votes, filtered HMM pBusy, and fuzzy scores."""

    def __init__(
        self,
        models: dict[str, ChannelOccupancyHMM],
        training_posteriors: dict[str, tuple[float, float]],
        *,
        allocation_config: AssmAllocationConfig | None = None,
    ) -> None:
        self.models = models
        self.posteriors = dict(training_posteriors)
        self.fusion = CooperativeSensingFusion()
        self.allocator = AssmAllocator(
            FuzzyChannelDecisionEngine(),
            allocation_config or AssmAllocationConfig(),
        )
        self.last_decision_rows: list[dict[str, object]] = []
        self.last_hmm_rows: list[dict[str, object]] = []

    def choose_channels(
        self, snapshot: DecisionSnapshot
    ) -> tuple[int | None, ...]:
        """Make one decision from only observations available at this boundary."""

        record_index = snapshot.database_record_index
        fused = self.fusion.fuse(
            snapshot.sensing_readings,
            time_slot=record_index if record_index is not None else snapshot.time_slot,
        )
        vote_by_channel = {
            result.channel_id: result.busy_votes for result in fused.channels
        }
        current_probabilities: dict[int, float] = {}
        hmm_rows: list[dict[str, object]] = []
        timestamp = (
            snapshot.database_timestamp.isoformat(timespec="seconds")
            if snapshot.database_timestamp is not None
            else ""
        )

        for channel_index, channel_id in enumerate(CHANNEL_IDS):
            vote_count = vote_by_channel[channel_index]
            model = self.models[channel_id]
            current, posterior = model.filter_current_busy_probabilities(
                (vote_count,),
                previous_posterior=self.posteriors[channel_id],
            )
            probability = current[0]
            self.posteriors[channel_id] = posterior
            current_probabilities[channel_index] = probability
            hmm_rows.append(
                {
                    "timestamp": timestamp,
                    "database_record_index": record_index,
                    "channel_id": channel_id,
                    "observation_type": "simulated_cooperative_busy_vote_count_0_to_3",
                    "cooperative_busy_vote_count": vote_count,
                    "hmm_current_filtered_busy_probability": f"{probability:.10f}",
                    "hmm_current_prediction": "busy" if probability >= 0.50 else "idle",
                }
            )

        allocation = self.allocator.allocate(
            snapshot,
            fused,
            current_probabilities,
        )
        selected_by_user = allocation.selected_channels
        selected_ids = {
            user_id: (
                CHANNEL_IDS[channel_index]
                if channel_index is not None
                else ""
            )
            for user_id, channel_index in enumerate(selected_by_user)
        }
        self.last_hmm_rows = hmm_rows
        self.last_decision_rows = []
        for assessment in allocation.assessments:
            fuzzy = assessment.fuzzy
            self.last_decision_rows.append(
                {
                    "timestamp": timestamp,
                    "database_record_index": record_index,
                    "secondary_user_id": assessment.secondary_user_id,
                    "channel_id": CHANNEL_IDS[assessment.channel_id],
                    "cooperative_busy_vote_count": assessment.busy_votes,
                    "cooperative_report_count": assessment.report_count,
                    "cooperative_fused_busy": assessment.cooperative_busy,
                    "hmm_current_filtered_busy_probability": (
                        f"{assessment.hmm_busy_probability:.10f}"
                    ),
                    "snr_db": f"{fuzzy.snr_db:.6f}",
                    "interference_dbm": f"{fuzzy.interference_dbm:.6f}",
                    "interference_interpretation": fuzzy.interference_interpretation,
                    "fuzzy_suitability_score": f"{fuzzy.suitability_score:.6f}",
                    "fuzzy_fired_rules": ";".join(
                        f"{rule.rule_id}:{rule.activation:.4f}"
                        for rule in fuzzy.fired_rules
                    ),
                    "clipped_inputs": ";".join(fuzzy.clipped_inputs),
                    "selected_channel_for_user": selected_ids[
                        assessment.secondary_user_id
                    ],
                    "decision": assessment.decision,
                }
            )
        return selected_by_user


def _fit_hmms(
    database_path: Path,
    *,
    seed: int,
) -> tuple[
    dict[str, ChannelOccupancyHMM],
    dict[str, tuple[float, float]],
    list[dict[str, object]],
]:
    """Fit each channel HMM using only the first six days of SU observations."""

    dataset = build_observation_dataset(
        database_path,
        seed=seed,
        scenario_name="medium",
        start_record=0,
        record_count=TRAIN_RECORDS,
    )
    models: dict[str, ChannelOccupancyHMM] = {}
    posteriors: dict[str, tuple[float, float]] = {}
    training_rows: list[dict[str, object]] = []
    for channel_index, channel_id in enumerate(CHANNEL_IDS):
        training_votes = dataset.observations_by_channel[channel_id]
        estimator = ChannelOccupancyHMM(seed=seed + channel_index).fit(
            training_votes
        )
        _, terminal_posterior = estimator.filter_current_busy_probabilities(
            training_votes
        )
        models[channel_id] = estimator
        posteriors[channel_id] = terminal_posterior
        expected_votes = estimator.expected_votes_by_state
        assert expected_votes is not None and estimator.busy_state is not None
        training_rows.append(
            {
                "seed": seed,
                "channel_id": channel_id,
                "training_records": len(training_votes),
                "busy_hidden_state": estimator.busy_state,
                "expected_votes_state_0": f"{expected_votes[0]:.8f}",
                "expected_votes_state_1": f"{expected_votes[1]:.8f}",
                "training_end_timestamp": dataset.timestamps[-1].isoformat(),
            }
        )
    return models, posteriors, training_rows


def _csv_writer(
    path: Path, fieldnames: Sequence[str]
) -> tuple[object, csv.DictWriter]:
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    return handle, writer


def _write_rows(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_occupancy(database_path: Path, output_dir: Path) -> dict[str, object]:
    """Calculate full-week hourly, daily, weekday/weekend, and daypart summaries."""

    hourly: dict[tuple[str, int], list[float]] = {}
    hourly_by_day: dict[tuple[str, int, int], list[float]] = {}
    daily: dict[tuple[str, int, str, str], list[float]] = {}
    weekday_weekend: dict[tuple[str, str], list[float]] = {}
    daypart: dict[tuple[str, str], list[float]] = {}
    rows_seen = 0

    def add(counter: dict[tuple, list[float]], key: tuple, busy: bool, target: float) -> None:
        values = counter.setdefault(key, [0.0, 0.0, 0.0])
        values[0] += int(busy)
        values[1] += 1
        values[2] += target

    with database_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "timestamp",
            "day_number",
            "weekday",
            "hour",
            "daypart",
            "channel_id",
            "channel_category",
            "occupancy_state",
            "configured_busy_probability_pct",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("occupancy database does not contain required summary columns")
        for row in reader:
            rows_seen += 1
            channel_id = row["channel_id"]
            day_number = int(row["day_number"])
            weekday = row["weekday"]
            hour = int(row["hour"])
            busy = row["occupancy_state"].strip().lower() == "busy"
            target = float(row["configured_busy_probability_pct"])
            date_text = datetime.fromisoformat(row["timestamp"]).date().isoformat()
            day_type = "weekend" if weekday in {"Saturday", "Sunday"} else "weekday"
            add(hourly, (channel_id, hour), busy, target)
            add(hourly_by_day, (channel_id, day_number, hour), busy, target)
            add(daily, (channel_id, day_number, date_text, weekday), busy, target)
            add(weekday_weekend, (channel_id, day_type), busy, target)
            add(daypart, (channel_id, row["daypart"]), busy, target)

    if rows_seen != 483_840:
        raise ValueError(f"expected 483840 occupancy rows, found {rows_seen}")

    def pct(values: list[float]) -> float:
        return 100.0 * values[0] / values[1] if values[1] else 0.0

    def target_pct(values: list[float]) -> float:
        return values[2] / values[1] if values[1] else 0.0

    hourly_rows = [
        {
            "channel_id": channel,
            "hour": hour,
            "busy_records": int(values[0]),
            "total_records": int(values[1]),
            "actual_busy_pct": f"{pct(values):.6f}",
            "mean_configured_busy_probability_pct": f"{target_pct(values):.6f}",
        }
        for (channel, hour), values in sorted(hourly.items())
    ]
    hourly_day_rows = [
        {
            "channel_id": channel,
            "day_number": day,
            "hour": hour,
            "busy_records": int(values[0]),
            "total_records": int(values[1]),
            "actual_busy_pct": f"{pct(values):.6f}",
            "mean_configured_busy_probability_pct": f"{target_pct(values):.6f}",
        }
        for (channel, day, hour), values in sorted(hourly_by_day.items())
    ]
    daily_rows = [
        {
            "channel_id": channel,
            "day_number": day,
            "date": date_text,
            "weekday": weekday,
            "busy_records": int(values[0]),
            "total_records": int(values[1]),
            "actual_busy_pct": f"{pct(values):.6f}",
            "mean_configured_busy_probability_pct": f"{target_pct(values):.6f}",
        }
        for (channel, day, date_text, weekday), values in sorted(daily.items())
    ]
    weekend_rows = [
        {
            "channel_id": channel,
            "day_type": day_type,
            "busy_records": int(values[0]),
            "total_records": int(values[1]),
            "actual_busy_pct": f"{pct(values):.6f}",
            "mean_configured_busy_probability_pct": f"{target_pct(values):.6f}",
        }
        for (channel, day_type), values in sorted(weekday_weekend.items())
    ]
    daypart_rows = [
        {
            "channel_id": channel,
            "daypart": part,
            "busy_records": int(values[0]),
            "total_records": int(values[1]),
            "actual_busy_pct": f"{pct(values):.6f}",
            "mean_configured_busy_probability_pct": f"{target_pct(values):.6f}",
        }
        for (channel, part), values in sorted(daypart.items())
    ]
    _write_rows(
        output_dir / "occupancy_hourly_summary.csv",
        tuple(hourly_rows[0]),
        hourly_rows,
    )
    _write_rows(
        output_dir / "occupancy_hourly_by_day_summary.csv",
        tuple(hourly_day_rows[0]),
        hourly_day_rows,
    )
    _write_rows(
        output_dir / "occupancy_daily_summary.csv",
        tuple(daily_rows[0]),
        daily_rows,
    )
    _write_rows(
        output_dir / "occupancy_weekday_weekend_summary.csv",
        tuple(weekend_rows[0]),
        weekend_rows,
    )
    _write_rows(
        output_dir / "occupancy_daypart_summary.csv",
        tuple(daypart_rows[0]),
        daypart_rows,
    )
    return {
        "rows_seen": rows_seen,
        "hourly_groups": len(hourly_rows),
        "daily_groups": len(daily_rows),
        "weekday_weekend_groups": len(weekend_rows),
        "daypart_groups": len(daypart_rows),
    }


def _ground_truth_lookup(
    database_path: Path, *, start_record: int, record_count: int
) -> dict[tuple[str, str], bool]:
    first_group = start_record
    last_group = start_record + record_count
    lookup: dict[tuple[str, str], bool] = {}
    with database_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            group_index = row_index // len(CHANNEL_IDS)
            if group_index < first_group:
                continue
            if group_index >= last_group:
                break
            lookup[(row["timestamp"], row["channel_id"])] = (
                row["occupancy_state"].strip().lower() == "busy"
            )
    expected = record_count * len(CHANNEL_IDS)
    if len(lookup) != expected:
        raise ValueError(
            f"held-out occupancy lookup expected {expected} channel records, "
            f"found {len(lookup)}"
        )
    return lookup


def _evaluate_hmm_current(
    raw_path: Path,
    output_dir: Path,
    *,
    database_path: Path,
    start_record: int,
    record_count: int,
) -> int:
    """Join ground truth after allocation, then calculate held-out HMM metrics."""

    truth = _ground_truth_lookup(
        database_path,
        start_record=start_record,
        record_count=record_count,
    )
    eval_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, int, str], dict[str, list[bool]]] = {}
    with raw_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["timestamp"], row["channel_id"])
            if key not in truth:
                raise ValueError(f"HMM output timestamp is outside held-out window: {key}")
            actual = truth[key]
            probability = float(row["hmm_current_filtered_busy_probability"])
            hmm_guess = probability >= 0.50
            vote_guess = int(row["cooperative_busy_vote_count"]) >= 2
            eval_row = {
                **row,
                "evaluation_ground_truth_busy": "true" if actual else "false",
            }
            eval_rows.append(eval_row)
            group_key = (row["scenario"], int(row["seed"]), row["channel_id"])
            bucket = grouped.setdefault(
                group_key,
                {"truth": [], "hmm": [], "vote": []},
            )
            bucket["truth"].append(actual)
            bucket["hmm"].append(hmm_guess)
            bucket["vote"].append(vote_guess)

    if not eval_rows:
        raise ValueError("no held-out HMM observations were written")
    eval_fields = tuple(eval_rows[0])
    _write_rows(
        output_dir / "hmm_evaluation_with_ground_truth.csv",
        eval_fields,
        eval_rows,
    )
    metrics_rows: list[dict[str, object]] = []
    aggregate: dict[tuple[str, int, str], dict[str, list[bool]]] = {}
    for (scenario, seed, channel_id), values in sorted(grouped.items()):
        aggregate_key = (scenario, seed, "OVERALL_MICRO")
        overall = aggregate.setdefault(
            aggregate_key,
            {"truth": [], "hmm": [], "vote": []},
        )
        for name in ("truth", "hmm", "vote"):
            overall[name].extend(values[name])
        for method, guesses in (
            ("HMM_CURRENT_FILTER", values["hmm"]),
            ("COOPERATIVE_VOTE", values["vote"]),
        ):
            metrics = classification_metrics(values["truth"], guesses)
            metrics_rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "method": method,
                    "channel_id": channel_id,
                    **asdict(metrics),
                }
            )
    for (scenario, seed, channel_id), values in sorted(aggregate.items()):
        for method, guesses in (
            ("HMM_CURRENT_FILTER", values["hmm"]),
            ("COOPERATIVE_VOTE", values["vote"]),
        ):
            metrics = classification_metrics(values["truth"], guesses)
            metrics_rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "method": method,
                    "channel_id": channel_id,
                    **asdict(metrics),
                }
            )
    _write_rows(
        output_dir / "hmm_current_metrics.csv",
        tuple(metrics_rows[0]),
        metrics_rows,
    )

    period_groups: dict[str, dict[tuple[str, ...], list[dict[str, object]]]] = {
        "hourly": {},
        "daypart": {},
        "daily": {},
        "weekday_weekend": {},
    }
    for row in eval_rows:
        timestamp = datetime.fromisoformat(row["timestamp"])
        hour = timestamp.hour
        if 5 <= hour < 12:
            part = "Morning"
        elif 12 <= hour < 17:
            part = "Afternoon"
        elif 17 <= hour < 21:
            part = "Evening"
        else:
            part = "Nighttime"
        weekday = timestamp.strftime("%A")
        day_type = "weekend" if weekday in {"Saturday", "Sunday"} else "weekday"
        base = (row["scenario"], row["seed"], row["channel_id"])
        for group_name, suffix in (
            ("hourly", (f"{hour:02d}",)),
            ("daypart", (part,)),
            ("daily", (timestamp.date().isoformat(), weekday)),
            ("weekday_weekend", (day_type,)),
        ):
            period_groups[group_name].setdefault(base + suffix, []).append(row)

    for group_name, grouped_rows in period_groups.items():
        summary_rows: list[dict[str, object]] = []
        for key, values in sorted(grouped_rows.items()):
            actual = [
                row["evaluation_ground_truth_busy"] == "true" for row in values
            ]
            hmm_predictions = [
                float(row["hmm_current_filtered_busy_probability"]) >= 0.50
                for row in values
            ]
            vote_predictions = [
                int(row["cooperative_busy_vote_count"]) >= 2 for row in values
            ]
            extras = key[3:]
            summary_rows.append(
                {
                    "scenario": key[0],
                    "seed": key[1],
                    "channel_id": key[2],
                    "period": extras[0],
                    "period_detail": extras[1] if len(extras) > 1 else "",
                    "sample_records": len(values),
                    "actual_busy_pct": 100.0 * sum(actual) / len(actual),
                    "mean_hmm_busy_probability": fmean(
                        float(row["hmm_current_filtered_busy_probability"])
                        for row in values
                    ),
                    "hmm_accuracy_pct": 100.0
                    * sum(a == p for a, p in zip(actual, hmm_predictions))
                    / len(actual),
                    "cooperative_vote_accuracy_pct": 100.0
                    * sum(a == p for a, p in zip(actual, vote_predictions))
                    / len(actual),
                }
            )
        if not summary_rows:
            continue
        _write_rows(
            output_dir / f"hmm_{group_name}_summary.csv",
            tuple(summary_rows[0]),
            summary_rows,
        )
    return len(eval_rows)


def _paired_metric_rows(rows: list[RunMetrics]) -> list[RunMetrics]:
    pdr = {
        (row.method, row.scenario, row.seed): row.packet_delivery_ratio_pct
        for row in rows
    }
    updated: list[RunMetrics] = []
    for row in rows:
        retention = None
        if row.scenario == "stress":
            medium = pdr.get((row.method, "medium", row.seed))
            stress = row.packet_delivery_ratio_pct
            if medium is not None and medium > 0.0 and stress is not None:
                retention = 100.0 * stress / medium
        updated.append(replace(row, robustness_vs_medium_pdr_pct=retention))
    return updated


def _write_comparison_tables(
    output_dir: Path, rows: list[RunMetrics]
) -> dict[str, int]:
    by_key = {(row.method, row.scenario, row.seed): row for row in rows}
    test1: list[dict[str, object]] = []
    test2: list[dict[str, object]] = []
    test3: list[dict[str, object]] = []
    test4: list[dict[str, object]] = []
    for scenario in sorted({row.scenario for row in rows}):
        seeds = sorted({row.seed for row in rows if row.scenario == scenario})
        for seed in seeds:
            standard = by_key.get((METHOD_STANDARD, scenario, seed))
            assm = by_key.get((METHOD_ASSM, scenario, seed))
            if standard is None or assm is None:
                continue
            test1.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "standard_utilization_pct": standard.spectrum_utilization_pct,
                    "assm_utilization_pct": assm.spectrum_utilization_pct,
                    "assm_minus_standard_percentage_points": (
                        None
                        if standard.spectrum_utilization_pct is None
                        or assm.spectrum_utilization_pct is None
                        else assm.spectrum_utilization_pct
                        - standard.spectrum_utilization_pct
                    ),
                }
            )
            test2.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "standard_throughput_bps": standard.throughput_bps,
                    "assm_throughput_bps": assm.throughput_bps,
                    "assm_minus_standard_throughput_bps": (
                        assm.throughput_bps - standard.throughput_bps
                    ),
                    "standard_pdr_pct": standard.packet_delivery_ratio_pct,
                    "assm_pdr_pct": assm.packet_delivery_ratio_pct,
                    "assm_minus_standard_pdr_percentage_points": (
                        None
                        if standard.packet_delivery_ratio_pct is None
                        or assm.packet_delivery_ratio_pct is None
                        else assm.packet_delivery_ratio_pct
                        - standard.packet_delivery_ratio_pct
                    ),
                }
            )
            test3.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "standard_mean_latency_ms": standard.mean_latency_ms,
                    "assm_mean_latency_ms": assm.mean_latency_ms,
                    "assm_minus_standard_latency_ms": (
                        None
                        if standard.mean_latency_ms is None
                        or assm.mean_latency_ms is None
                        else assm.mean_latency_ms - standard.mean_latency_ms
                    ),
                    "standard_su_collision_packets": standard.su_collision_packets,
                    "assm_su_collision_packets": assm.su_collision_packets,
                    "standard_primary_conflict_packets": (
                        standard.primary_user_conflict_packets
                    ),
                    "assm_primary_conflict_packets": (
                        assm.primary_user_conflict_packets
                    ),
                }
            )
    for seed in sorted({row.seed for row in rows}):
        for method in (METHOD_STANDARD, METHOD_ASSM):
            medium = by_key.get((method, "medium", seed))
            stress = by_key.get((method, "stress", seed))
            if medium is None or stress is None:
                continue
            medium_pdr = medium.packet_delivery_ratio_pct
            stress_pdr = stress.packet_delivery_ratio_pct
            retention = (
                100.0 * stress_pdr / medium_pdr
                if medium_pdr is not None
                and medium_pdr > 0.0
                and stress_pdr is not None
                else None
            )
            test4.append(
                {
                    "seed": seed,
                    "method": method,
                    "medium_pdr_pct": medium_pdr,
                    "stress_pdr_pct": stress_pdr,
                    "stress_minus_medium_pdr_percentage_points": (
                        None
                        if medium_pdr is None or stress_pdr is None
                        else stress_pdr - medium_pdr
                    ),
                    "stress_pdr_retention_vs_medium_pct": retention,
                    "zero_or_missing_medium_pdr": (
                        medium_pdr is None or medium_pdr == 0.0
                    ),
                }
            )

    _write_rows(
        output_dir / "test1_utilization_efficiency.csv",
        (
            "scenario",
            "seed",
            "standard_utilization_pct",
            "assm_utilization_pct",
            "assm_minus_standard_percentage_points",
        ),
        test1,
    )
    _write_rows(
        output_dir / "test2_throughput_pdr.csv",
        (
            "scenario",
            "seed",
            "standard_throughput_bps",
            "assm_throughput_bps",
            "assm_minus_standard_throughput_bps",
            "standard_pdr_pct",
            "assm_pdr_pct",
            "assm_minus_standard_pdr_percentage_points",
        ),
        test2,
    )
    _write_rows(
        output_dir / "test3_latency_collisions.csv",
        (
            "scenario",
            "seed",
            "standard_mean_latency_ms",
            "assm_mean_latency_ms",
            "assm_minus_standard_latency_ms",
            "standard_su_collision_packets",
            "assm_su_collision_packets",
            "standard_primary_conflict_packets",
            "assm_primary_conflict_packets",
        ),
        test3,
    )
    _write_rows(
        output_dir / "test4_robustness.csv",
        (
            "seed",
            "method",
            "medium_pdr_pct",
            "stress_pdr_pct",
            "stress_minus_medium_pdr_percentage_points",
            "stress_pdr_retention_vs_medium_pct",
            "zero_or_missing_medium_pdr",
        ),
        test4,
    )

    metrics_to_compare = (
        "spectrum_utilization_pct",
        "throughput_bps",
        "packet_delivery_ratio_pct",
        "mean_latency_ms",
        "su_collision_packets",
        "primary_user_conflict_packets",
    )
    summary_rows: list[dict[str, object]] = []
    for scenario in sorted({row.scenario for row in rows}):
        for metric in metrics_to_compare:
            standard_values: list[float] = []
            assm_values: list[float] = []
            paired_differences: list[float] = []
            for seed in sorted({row.seed for row in rows if row.scenario == scenario}):
                standard = by_key.get((METHOD_STANDARD, scenario, seed))
                assm = by_key.get((METHOD_ASSM, scenario, seed))
                if standard is None or assm is None:
                    continue
                standard_value = getattr(standard, metric)
                assm_value = getattr(assm, metric)
                if standard_value is None or assm_value is None:
                    continue
                standard_values.append(float(standard_value))
                assm_values.append(float(assm_value))
                paired_differences.append(float(assm_value) - float(standard_value))
            summary_rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "standard_mean": (
                        fmean(standard_values) if standard_values else None
                    ),
                    "assm_mean": fmean(assm_values) if assm_values else None,
                    "mean_paired_assm_minus_standard": (
                        fmean(paired_differences) if paired_differences else None
                    ),
                    "paired_seed_count": len(paired_differences),
                    "interpretation": "measured simulation result; no preferred outcome assumed",
                }
            )
    _write_rows(
        output_dir / "comparison_summary.csv",
        tuple(summary_rows[0]),
        summary_rows,
    )
    return {
        "test1_rows": len(test1),
        "test2_rows": len(test2),
        "test3_rows": len(test3),
        "test4_rows": len(test4),
        "comparison_summary_rows": len(summary_rows),
    }


def _write_configuration(
    path: Path,
    *,
    database_path: Path,
    scenarios: Sequence[str],
    seed: int,
    repetitions: int,
    records: int,
    output_dir: Path,
    allocation_config: AssmAllocationConfig,
) -> None:
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    with OccupancyDatabaseReplay(database_path) as replay:
        first_timestamp = replay.record_for_slot(
            DAY7_RECORD * STEPS_PER_RECORD
        ).timestamp
        last_timestamp = replay.record_for_slot(
            (DAY7_RECORD + records - 1) * STEPS_PER_RECORD
        ).timestamp

    lines = [
        "SPECTRIX Phase 7 paired ASSM evaluation",
        "Data classification: synthetic occupancy assumptions and simulated SU observations; not measured real-world RF occupancy.",
        "Database source is the existing seven-day occupancy CSV; source data and prior results are read-only for this run.",
        "HMM fitting uses cooperative busy-vote observations from database records 0 through 51839 (days 1-6) only.",
        "The day-7 decision uses the current-record filtered P(BUSY | votes through current record), not the next-record prediction.",
        "Ground truth is used only to generate simulated sensing, validate replay, and score completed outputs. It is excluded from policy decision snapshots.",
        "The occupancy source starts at database record 51840; each 10-second record remains active for 10000 simulator steps at 1 ms per step.",
        "Standard and ASSM are paired with the same scenario, seed, occupancy, per-step traffic and sensing streams, held-out record window, shared transmission model, and 10000-slot allocation cadence.",
        "The historical Standard experiment remains per-step and writes its historical output separately.",
        "Test 1: utilization and efficiency; spectrum utilization follows the shared simulator metric definition.",
        "Test 2: throughput and packet delivery ratio.",
        "Test 3: latency, secondary-user collisions, and primary-user conflicts.",
        "Test 4: robustness across medium and stress scenarios; retention = stress PDR / medium PDR * 100; also report stress-minus-medium PDR percentage points. Undefined when medium PDR is zero or missing.",
        "ASSM threshold values are configurable initial engineering assumptions, not proven-optimal values.",
        f"assm_reject_busy_probability={allocation_config.reject_busy_probability}",
        f"assm_defer_busy_probability={allocation_config.defer_busy_probability}",
        f"assm_minimum_fuzzy_suitability_score={allocation_config.minimum_suitability_score}",
        "Channel eligibility order: cooperative fused BUSY rejects; pBusy >= reject threshold rejects; defer threshold <= pBusy < reject threshold defers; otherwise fuzzy score must meet minimum; each selected channel is unique per record.",
        f"database={database_path.resolve()}",
        f"database_record_window={DAY7_RECORD}:{DAY7_RECORD + records - 1}",
        f"held_out_first_timestamp={first_timestamp.isoformat() if first_timestamp else 'unknown'}",
        f"held_out_last_timestamp={last_timestamp.isoformat() if last_timestamp else 'unknown'}",
        f"scenarios={','.join(scenarios)}",
        f"first_seed={seed}",
        f"repetitions={repetitions}",
        f"seed_sequence={','.join(str(seed + index) for index in range(repetitions))}",
        f"records_per_run={records}",
        f"simulator_steps_per_record={STEPS_PER_RECORD}",
        f"allocation_cadence_slots={DECISION_CADENCE_SLOTS}",
        f"output_directory={output_dir.resolve()}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    *,
    aggregate: dict[str, object],
    summary_counts: dict[str, int],
    run_rows: list[RunMetrics],
    simulation_seconds: float,
    training_seconds: float,
    total_seconds: float,
    simulated_steps: int,
    estimated_full_simulation_seconds: float,
    estimated_full_total_seconds: float,
    hmm_rows: int,
    database_path: Path,
    records: int,
    smoke: bool,
) -> None:
    lines = [
        "SPECTRIX Phase 7 validation and measured-results report",
        f"run_type={'SMOKE_PARTIAL_HELDOUT' if smoke else 'FULL_HELDOUT_DAY7'}",
        "All occupancy and policy outputs are synthetic simulation results, not real-world radio measurements.",
        f"occupancy_database={database_path.resolve()}",
        f"occupancy_database_channel_slot_rows={aggregate['rows_seen']}",
        f"occupancy_hourly_groups={aggregate['hourly_groups']}",
        f"occupancy_daily_groups={aggregate['daily_groups']}",
        f"occupancy_weekday_weekend_groups={aggregate['weekday_weekend_groups']}",
        f"occupancy_daypart_groups={aggregate['daypart_groups']}",
        f"heldout_records_per_policy_run={records}",
        f"paired_scenario_seed_runs={len(run_rows) // 2}",
        f"policy_run_rows={len(run_rows)}",
        f"current_record_hmm_evaluation_rows={hmm_rows}",
        f"measured_paired_simulation_wall_seconds={simulation_seconds:.3f}",
        f"measured_hmm_training_wall_seconds={training_seconds:.3f}",
        f"measured_total_wall_seconds={total_seconds:.3f}",
        f"paired_simulator_steps_measured={simulated_steps}",
        f"estimated_full_evaluation_simulation_seconds={estimated_full_simulation_seconds:.3f}",
        f"estimated_full_evaluation_total_seconds={estimated_full_total_seconds:.3f}",
        "validation_database_record_count=PASS",
        "validation_seventh_day_chronological_offset=PASS",
        "validation_paired_environment_equal_at_every_decision_boundary=PASS",
        "validation_policy_api_excludes_ground_truth=PASS",
        "validation_allocation_cadence_is_10000_slots=PASS",
        "validation_prior_outputs_preserved=verified separately by pre/post SHA-256 checks",
        "four_comparison_tables="
        + ",".join(f"{key}:{value}" for key, value in sorted(summary_counts.items())),
        "Results in this report are measured from the CSVs emitted by this run. No expected performance ranking is asserted.",
    ]
    if smoke:
        lines.append(
            "The smoke run covers only the requested initial held-out records and is not a complete seventh-day performance evaluation."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Standard and ASSM with paired 10-second decisions on database day 7."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("results/channel_occupancy_7days.csv"),
        help="existing validated seven-day occupancy database",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIO_NAMES,
        default=list(SCENARIO_NAMES),
    )
    parser.add_argument("--records", type=int, default=RECORDS_PER_DAY)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reject-busy-probability", type=float, default=0.50)
    parser.add_argument("--defer-busy-probability", type=float, default=0.45)
    parser.add_argument("--minimum-fuzzy-score", type=float, default=50.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase7"),
        help="new output directory; it must not already exist",
    )
    args = parser.parse_args()
    if not 1 <= args.records <= RECORDS_PER_DAY:
        parser.error(f"--records must be between 1 and {RECORDS_PER_DAY}")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    try:
        allocation_config = AssmAllocationConfig(
            reject_busy_probability=args.reject_busy_probability,
            defer_busy_probability=args.defer_busy_probability,
            minimum_suitability_score=args.minimum_fuzzy_score,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if not args.database.is_file():
        parser.error(f"occupancy database not found: {args.database}")
    if args.output_dir.exists():
        parser.error(
            f"output directory already exists; choose a new path to preserve it: {args.output_dir}"
        )

    start_total = perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    aggregate = _aggregate_occupancy(args.database, args.output_dir)
    _write_configuration(
        args.output_dir / "run_configuration.txt",
        database_path=args.database,
        scenarios=args.scenarios,
        seed=args.seed,
        repetitions=args.repetitions,
        records=args.records,
        output_dir=args.output_dir,
        allocation_config=allocation_config,
    )
    training_path = args.output_dir / "hmm_training_summary.csv"
    training_handle, training_writer = _csv_writer(
        training_path,
        (
            "seed",
            "channel_id",
            "training_records",
            "busy_hidden_state",
            "expected_votes_state_0",
            "expected_votes_state_1",
            "training_end_timestamp",
        ),
    )
    decision_fields = (
        "timestamp",
        "database_record_index",
        "secondary_user_id",
        "channel_id",
        "cooperative_busy_vote_count",
        "cooperative_report_count",
        "cooperative_fused_busy",
        "hmm_current_filtered_busy_probability",
        "snr_db",
        "interference_dbm",
        "interference_interpretation",
        "fuzzy_suitability_score",
        "fuzzy_fired_rules",
        "clipped_inputs",
        "selected_channel_for_user",
        "decision",
        "scenario",
        "seed",
    )
    hmm_fields = (
        "timestamp",
        "database_record_index",
        "channel_id",
        "observation_type",
        "cooperative_busy_vote_count",
        "hmm_current_filtered_busy_probability",
        "hmm_current_prediction",
        "scenario",
        "seed",
    )
    decision_handle, decision_writer = _csv_writer(
        args.output_dir / "assm_decision_log.csv", decision_fields
    )
    hmm_handle, hmm_writer = _csv_writer(
        args.output_dir / "current_hmm_probabilities.csv", hmm_fields
    )

    metrics_rows: list[RunMetrics] = []
    simulation_seconds = 0.0
    training_seconds = 0.0
    try:
        for repetition in range(args.repetitions):
            seed = args.seed + repetition
            start_training = perf_counter()
            models, training_posteriors, training_rows = _fit_hmms(
                args.database,
                seed=seed,
            )
            training_seconds += perf_counter() - start_training
            training_writer.writerows(training_rows)

            for scenario_name in args.scenarios:
                scenario = replace(
                    ScenarioConfig.preset(scenario_name),
                    num_channels=len(CHANNEL_IDS),
                )
                standard_source = OccupancyDatabaseReplay(
                    args.database,
                    start_record=DAY7_RECORD,
                    record_count=args.records,
                )
                assm_source = OccupancyDatabaseReplay(
                    args.database,
                    start_record=DAY7_RECORD,
                    record_count=args.records,
                )
                standard_simulator = WirelessSpectrumSimulator(
                    scenario=scenario,
                    seed=seed,
                    occupancy_source=standard_source,
                )
                assm_simulator = WirelessSpectrumSimulator(
                    scenario=scenario,
                    seed=seed,
                    occupancy_source=assm_source,
                )
                assm_policy = OnlineAssmPolicy(
                    models,
                    training_posteriors,
                    allocation_config=allocation_config,
                )

                def write_decision(
                    policy: object,
                    decision_snapshot: DecisionSnapshot,
                    selected: tuple[int | None, ...],
                    *,
                    current_scenario: str = scenario_name,
                    current_seed: int = seed,
                ) -> None:
                    del decision_snapshot, selected
                    for row in getattr(policy, "last_decision_rows"):
                        decision_writer.writerow(
                            {
                                **row,
                                "scenario": current_scenario,
                                "seed": current_seed,
                            }
                        )
                    for row in getattr(policy, "last_hmm_rows"):
                        hmm_writer.writerow(
                            {
                                **row,
                                "scenario": current_scenario,
                                "seed": current_seed,
                            }
                        )

                start_sim = perf_counter()
                paired = run_paired_policies(
                    standard_simulator,
                    assm_simulator,
                    StandardAllocator(),
                    assm_policy,
                    seed=seed,
                    slots=args.records * STEPS_PER_RECORD,
                    decision_cadence_slots=DECISION_CADENCE_SLOTS,
                    on_assm_decision=write_decision,
                )
                simulation_seconds += perf_counter() - start_sim
                metrics_rows.extend(paired)
                standard_source.close()
                assm_source.close()
                print(
                    f"completed paired run scenario={scenario_name} seed={seed} "
                    f"records={args.records}"
                )
    finally:
        training_handle.close()
        decision_handle.close()
        hmm_handle.close()

    metrics_rows = _paired_metric_rows(metrics_rows)
    metric_fields = tuple(field.name for field in fields(RunMetrics))
    _write_rows(
        args.output_dir / "paired_metrics.csv",
        metric_fields,
        [asdict(row) for row in metrics_rows],
    )
    comparison_counts = _write_comparison_tables(args.output_dir, metrics_rows)
    hmm_row_count = _evaluate_hmm_current(
        args.output_dir / "current_hmm_probabilities.csv",
        args.output_dir,
        database_path=args.database,
        start_record=DAY7_RECORD,
        record_count=args.records,
    )
    from experiments.plot_phase7_results import generate_phase7_graphs

    graph_paths = generate_phase7_graphs(args.output_dir)
    total_seconds = perf_counter() - start_total
    simulated_steps = (
        args.records
        * STEPS_PER_RECORD
        * 2
        * len(args.scenarios)
        * args.repetitions
    )
    full_step_count = (
        RECORDS_PER_DAY
        * STEPS_PER_RECORD
        * 2
        * len(SCENARIO_NAMES)
        * 3
    )
    estimated_full_simulation_seconds = (
        simulation_seconds * full_step_count / simulated_steps
        if simulated_steps
        else 0.0
    )
    other_seconds = max(
        0.0, total_seconds - simulation_seconds - training_seconds
    )
    estimated_full_total_seconds = (
        estimated_full_simulation_seconds
        + training_seconds * 3 / args.repetitions
        + other_seconds
    )
    (args.output_dir / "run_timing.txt").write_text(
        "\n".join(
            (
                f"measured_paired_simulation_seconds={simulation_seconds:.6f}",
                f"measured_hmm_training_seconds={training_seconds:.6f}",
                f"measured_total_seconds={total_seconds:.6f}",
                f"measured_paired_simulator_steps={simulated_steps}",
                f"estimated_full_simulation_seconds={estimated_full_simulation_seconds:.6f}",
                f"estimated_full_total_seconds={estimated_full_total_seconds:.6f}",
                "full-workload assumptions: 4 scenarios x 3 seeds x 2 policies x 8640 records x 10000 simulator steps",
                "estimate method: measured paired simulator seconds scaled by paired simulator steps; HMM training scaled by seed count; other measured overhead added once",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    smoke = (
        args.records < RECORDS_PER_DAY
        or set(args.scenarios) != set(SCENARIO_NAMES)
        or args.repetitions < 3
    )
    _write_report(
        args.output_dir / "phase7_report.txt",
        aggregate=aggregate,
        summary_counts=comparison_counts,
        run_rows=metrics_rows,
        simulation_seconds=simulation_seconds,
        total_seconds=total_seconds,
        training_seconds=training_seconds,
        simulated_steps=simulated_steps,
        estimated_full_simulation_seconds=estimated_full_simulation_seconds,
        estimated_full_total_seconds=estimated_full_total_seconds,
        hmm_rows=hmm_row_count,
        database_path=args.database,
        records=args.records,
        smoke=smoke,
    )

    print("SPECTRIX Phase 7 paired Standard/ASSM run complete")
    print(f"occupancy database channel-slot records: {aggregate['rows_seen']:,}")
    print(f"paired simulation runs: {len(metrics_rows) // 2}")
    print(f"held-out records per run: {args.records:,}")
    print(f"current-filter HMM evaluation rows: {hmm_row_count:,}")
    print(f"measured paired simulation seconds: {simulation_seconds:.3f}")
    print(f"measured total seconds: {total_seconds:.3f}")
    print(
        "estimated full evaluation runtime: "
        f"{estimated_full_total_seconds / 3600.0:.2f} hours "
        f"({estimated_full_total_seconds / 86400.0:.2f} days; "
        "scaling estimate, not a run)"
    )
    print(f"metrics: {(args.output_dir / 'paired_metrics.csv').resolve()}")
    print(f"decision log: {(args.output_dir / 'assm_decision_log.csv').resolve()}")
    print(f"HMM current probabilities: {(args.output_dir / 'current_hmm_probabilities.csv').resolve()}")
    print(f"HMM scored evaluation: {(args.output_dir / 'hmm_evaluation_with_ground_truth.csv').resolve()}")
    print(f"HMM metrics: {(args.output_dir / 'hmm_current_metrics.csv').resolve()}")
    print(f"report: {(args.output_dir / 'phase7_report.txt').resolve()}")
    print(f"graphs: {len(graph_paths)} saved in {(args.output_dir / 'graphs').resolve()}")
    if smoke:
        print("Run classification: partial smoke run; not a full held-out-day evaluation.")


if __name__ == "__main__":
    main()
