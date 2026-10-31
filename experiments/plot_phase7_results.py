"""Create figures from measured Phase 7 occupancy, HMM, and policy CSV outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save_heatmap(
    values: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xticklabels: list[str] | None = None,
    yticklabels: list[str] | None = None,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    image = axis.imshow(values, aspect="auto", interpolation="nearest", vmin=0, vmax=100)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if xticklabels is not None:
        stride = max(1, len(xticklabels) // 12)
        positions = list(range(0, len(xticklabels), stride))
        axis.set_xticks(positions, [xticklabels[index] for index in positions], rotation=45)
    if yticklabels is not None:
        axis.set_yticks(range(len(yticklabels)), yticklabels)
    fig.colorbar(image, ax=axis, label="Busy occupancy (%)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _group_mean(
    rows: list[dict[str, str]], key_fields: tuple[str, ...], value_field: str
) -> dict[tuple[str, ...], float]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        try:
            grouped[tuple(row[field] for field in key_fields)].append(
                float(row[value_field])
            )
        except (KeyError, TypeError, ValueError):
            continue
    return {
        key: sum(values) / len(values)
        for key, values in grouped.items()
        if values
    }


def generate_phase7_graphs(results_dir: str | Path) -> tuple[Path, ...]:
    """Generate PNGs using only files already measured by this Phase 7 run."""

    results_dir = Path(results_dir)
    graph_dir = results_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    channel_ids = [f"CH{number:02d}" for number in range(1, 9)]
    generated: list[Path] = []

    hourly_by_day = _rows(results_dir / "occupancy_hourly_by_day_summary.csv")
    weekly_matrix = np.full((8, 7 * 24), np.nan)
    for row in hourly_by_day:
        channel_index = channel_ids.index(row["channel_id"])
        x = (int(row["day_number"]) - 1) * 24 + int(row["hour"])
        weekly_matrix[channel_index, x] = float(row["actual_busy_pct"])
    path = graph_dir / "weekly_occupancy_heatmap.png"
    _save_heatmap(
        weekly_matrix,
        title="Seven-day synthetic channel occupancy",
        xlabel="Day and hour",
        ylabel="Channel",
        xticklabels=[f"D{day + 1} {hour:02d}:00" for day in range(7) for hour in range(24)],
        yticklabels=channel_ids,
        path=path,
    )
    generated.append(path)

    channel_hourly: dict[int, list[float]] = defaultdict(list)
    for row in hourly_by_day:
        hour_index = (int(row["day_number"]) - 1) * 24 + int(row["hour"])
        channel_hourly[hour_index].append(float(row["actual_busy_pct"]))
    weekly_hours = sorted(channel_hourly)
    weekly_means = [
        sum(channel_hourly[index]) / len(channel_hourly[index])
        for index in weekly_hours
    ]
    fig, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    axis.plot(weekly_hours, weekly_means, color="#276678")
    axis.set(
        xlabel="Hour index from start of seven-day dataset",
        ylabel="Mean busy occupancy across 8 channels (%)",
        title="Weekly synthetic occupancy profile",
    )
    path = graph_dir / "weekly_occupancy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    hourly = _rows(results_dir / "occupancy_hourly_summary.csv")
    hourly_matrix = np.full((8, 24), np.nan)
    for row in hourly:
        hourly_matrix[channel_ids.index(row["channel_id"]), int(row["hour"])] = float(
            row["actual_busy_pct"]
        )
    path = graph_dir / "hourly_occupancy.png"
    _save_heatmap(
        hourly_matrix,
        title="Synthetic occupancy by channel and hour (seven-day average)",
        xlabel="Hour of day",
        ylabel="Channel",
        xticklabels=[f"{hour:02d}:00" for hour in range(24)],
        yticklabels=channel_ids,
        path=path,
    )
    generated.append(path)

    daypart_rows = _rows(results_dir / "occupancy_daypart_summary.csv")
    parts = ["Morning", "Afternoon", "Evening", "Nighttime"]
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    width = 0.1
    x = np.arange(len(parts))
    for channel_index, channel in enumerate(channel_ids):
        lookup = {
            row["daypart"]: float(row["actual_busy_pct"])
            for row in daypart_rows
            if row["channel_id"] == channel
        }
        axis.bar(
            x + (channel_index - 3.5) * width,
            [lookup.get(part, 0.0) for part in parts],
            width,
            label=channel,
        )
    axis.set_title("Seven-day measured occupancy by daypart")
    axis.set_ylabel("Busy occupancy (%)")
    axis.set_xticks(x, parts)
    axis.legend(ncol=4, fontsize=8)
    path = graph_dir / "time_period_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    weekend = _rows(results_dir / "occupancy_weekday_weekend_summary.csv")
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    categories = ["weekday", "weekend"]
    x = np.arange(len(channel_ids))
    width = 0.36
    for offset, day_type in enumerate(categories):
        lookup = {
            row["channel_id"]: float(row["actual_busy_pct"])
            for row in weekend
            if row["day_type"] == day_type
        }
        axis.bar(
            x + (offset - 0.5) * width,
            [lookup.get(channel, 0.0) for channel in channel_ids],
            width,
            label=day_type.title(),
        )
    axis.set_title("Synthetic weekday/weekend occupancy")
    axis.set_ylabel("Busy occupancy (%)")
    axis.set_xticks(x, channel_ids)
    axis.legend()
    path = graph_dir / "weekend_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    hmm_rows = _rows(results_dir / "hmm_evaluation_with_ground_truth.csv")
    # Keep one paired seed/scenario for descriptive probability plots.
    if hmm_rows:
        first_scenario = hmm_rows[0]["scenario"]
        first_seed = hmm_rows[0]["seed"]
        descriptive = [
            row
            for row in hmm_rows
            if row["scenario"] == first_scenario and row["seed"] == first_seed
        ]
        probability_matrix = np.full((8, max(1, len(descriptive) // 8)), np.nan)
        by_channel: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for row in descriptive:
            by_channel[row["channel_id"]].append(
                (
                    row["timestamp"],
                    float(row["hmm_current_filtered_busy_probability"]),
                )
            )
        max_records = max((len(by_channel[channel]) for channel in channel_ids), default=1)
        probability_matrix = np.full((8, max_records), np.nan)
        time_labels: list[str] = []
        for channel_index, channel in enumerate(channel_ids):
            channel_values = sorted(by_channel[channel])
            for record_index, (timestamp, probability) in enumerate(channel_values):
                probability_matrix[channel_index, record_index] = probability * 100.0
                if channel_index == 0:
                    time_labels.append(timestamp[11:16])
        path = graph_dir / "hmm_channel_probability_heatmap.png"
        _save_heatmap(
            probability_matrix,
            title="Day-seven current-record HMM filtered busy probability",
            xlabel="Held-out 10-second records",
            ylabel="Channel",
            xticklabels=time_labels if len(time_labels) == max_records else None,
            yticklabels=channel_ids,
            path=path,
        )
        generated.append(path)

        bins = np.linspace(0.0, 1.0, 11)
        predicted: list[float] = []
        observed: list[float] = []
        for low, high in zip(bins[:-1], bins[1:]):
            in_bin = [
                row
                for row in descriptive
                if low <= float(row["hmm_current_filtered_busy_probability"])
                and (
                    float(row["hmm_current_filtered_busy_probability"]) < high
                    or high == 1.0
                )
            ]
            if in_bin:
                predicted.append(
                    sum(
                        float(row["hmm_current_filtered_busy_probability"])
                        for row in in_bin
                    )
                    / len(in_bin)
                )
                observed.append(
                    sum(row["evaluation_ground_truth_busy"] == "true" for row in in_bin)
                    / len(in_bin)
                )
        fig, axis = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        axis.plot(predicted, observed, marker="o", label="Measured bins")
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted P(BUSY)", ylabel="Observed busy fraction")
        axis.set_title(f"HMM probability calibration ({first_scenario}, seed {first_seed})")
        axis.legend()
        path = graph_dir / "hmm_probability_calibration.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

        hourly_accuracy: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for row in descriptive:
            hour = datetime.fromisoformat(row["timestamp"]).hour
            prediction = float(row["hmm_current_filtered_busy_probability"]) >= 0.5
            actual = row["evaluation_ground_truth_busy"] == "true"
            hourly_accuracy[hour][1] += 1
            hourly_accuracy[hour][0] += int(prediction == actual)
        hours = sorted(hourly_accuracy)
        fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        axis.plot(
            hours,
            [100.0 * hourly_accuracy[hour][0] / hourly_accuracy[hour][1] for hour in hours],
            marker="o",
        )
        axis.set(xlabel="Hour of day", ylabel="Accuracy (%)", ylim=(0, 100))
        axis.set_title("Day-seven current-filter HMM accuracy by hour")
        path = graph_dir / "hmm_time_period_accuracy.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

        daypart_rows = _rows(results_dir / "hmm_daypart_summary.csv")
        if daypart_rows:
            daypart_values: dict[str, list[float]] = defaultdict(list)
            for row in daypart_rows:
                if row["scenario"] == first_scenario and row["seed"] == first_seed:
                    daypart_values[row["period"]].append(float(row["hmm_accuracy_pct"]))
            labels = ["Morning", "Afternoon", "Evening", "Nighttime"]
            fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            axis.bar(
                labels,
                [
                    sum(daypart_values[label]) / len(daypart_values[label])
                    if daypart_values[label]
                    else 0.0
                    for label in labels
                ],
                color="#8064a2",
            )
            axis.set(ylabel="Accuracy (%)", ylim=(0, 100))
            axis.set_title("Day-seven current-filter HMM accuracy by daypart")
            path = graph_dir / "hmm_daypart_accuracy.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            generated.append(path)

    decisions = _rows(results_dir / "assm_decision_log.csv")
    if decisions:
        scores = [float(row["fuzzy_suitability_score"]) for row in decisions]
        fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        axis.hist(scores, bins=np.linspace(0, 100, 21), color="#3478a8", edgecolor="white")
        axis.set(xlabel="Fuzzy suitability score", ylabel="SU/channel decisions")
        axis.set_title("Measured ASSM fuzzy scores")
        path = graph_dir / "assm_fuzzy_score_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

        allocations: dict[str, int] = defaultdict(int)
        for row in decisions:
            if row["decision"] == "SELECTED":
                allocations[row["channel_id"]] += 1
        fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        axis.bar(channel_ids, [allocations[channel] for channel in channel_ids], color="#3c8d65")
        axis.set(xlabel="Channel", ylabel="Selected allocations")
        axis.set_title("ASSM selected-channel counts")
        path = graph_dir / "assm_channel_allocation_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

    comparison = _rows(results_dir / "comparison_summary.csv")
    scenarios = sorted({row["scenario"] for row in comparison})
    methods = ["standard_mean", "assm_mean"]
    colors = {"standard_mean": "#566573", "assm_mean": "#2874a6"}
    for metric, filename, title, ylabel in (
        ("spectrum_utilization_pct", "comparative_utilization.png", "Spectrum utilization", "Utilization (%)"),
        ("throughput_bps", "comparative_throughput.png", "Throughput", "Throughput (bit/s)"),
        ("packet_delivery_ratio_pct", "comparative_pdr.png", "Packet delivery ratio", "PDR (%)"),
        ("mean_latency_ms", "comparative_latency.png", "Mean latency", "Latency (ms)"),
        ("su_collision_packets", "comparative_su_collisions.png", "Secondary-user collisions", "Collision packets"),
        ("primary_user_conflict_packets", "comparative_primary_conflicts.png", "Primary-user conflicts", "Conflict packets"),
    ):
        relevant = [row for row in comparison if row["metric"] == metric]
        if not relevant:
            continue
        lookup = {row["scenario"]: row for row in relevant}
        x = np.arange(len(scenarios))
        width = 0.36
        fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for offset, method in enumerate(methods):
            values = [
                float(lookup[scenario][method])
                if scenario in lookup and lookup[scenario][method] not in {"", "None"}
                else 0.0
                for scenario in scenarios
            ]
            axis.bar(
                x + (offset - 0.5) * width,
                values,
                width,
                label="Standard" if offset == 0 else "ASSM",
                color=colors[method],
            )
        axis.set_xticks(x, scenarios)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend()
        path = graph_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

    robustness = _rows(results_dir / "test4_robustness.csv")
    if robustness:
        fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        methods_seen = ["STANDARD_10S", "ASSM"]
        for offset, method in enumerate(methods_seen):
            values = [
                float(row["stress_pdr_retention_vs_medium_pct"])
                for row in robustness
                if row["method"] == method
                and row["stress_pdr_retention_vs_medium_pct"] not in {"", "None"}
            ]
            if values:
                axis.bar(
                    offset,
                    sum(values) / len(values),
                    label=method,
                    color="#566573" if offset == 0 else "#2874a6",
                )
        axis.set_xticks(range(len(methods_seen)), ["Standard", "ASSM"])
        axis.set_ylabel("Stress PDR / medium PDR (%)")
        axis.set_title("Robustness retention across paired seeds")
        path = graph_dir / "comparative_robustness.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)
    else:
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.text(0.5, 0.5, "Requires paired medium and stress runs", ha="center", va="center")
        axis.set_axis_off()
        axis.set_title("Robustness test not available in this partial run")
        path = graph_dir / "comparative_robustness.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

    return tuple(generated)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate Phase 7 PNG graphs from an existing Phase 7 results directory."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/phase7"))
    args = parser.parse_args()
    paths = generate_phase7_graphs(args.results_dir)
    print(f"Generated {len(paths)} Phase 7 graphs in {(args.results_dir / 'graphs').resolve()}")


if __name__ == "__main__":
    main()
