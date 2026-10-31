"""Train and chronologically evaluate per-channel SPECTRIX occupancy HMMs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable

import hmmlearn

from partA.hmm_estimator import ChannelOccupancyHMM, ClassificationMetrics, classification_metrics
from partA.hmm_observation_pipeline import ObservationDataset, build_observation_dataset
from simulator.occupancy_database import CHANNEL_IDS


RECORDS_PER_DAY = 24 * 60 * 60 // 10
TRAIN_DAYS = 6
TEST_DAYS = 1
PREDICTIONS_FILENAME = "hmm_occupancy_predictions.csv"
METRICS_FILENAME = "hmm_metrics.csv"
CONFIGURATION_FILENAME = "hmm_run_configuration.txt"


def _csv_write(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _metric_row(
    method: str, channel_id: str, metrics: ClassificationMetrics
) -> dict[str, object]:
    return {"method": method, "channel_id": channel_id, **asdict(metrics)}


def evaluate_dataset(
    dataset: ObservationDataset,
    *,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Train on days 1–6 and score next-record predictions on day 7."""

    expected_records = (TRAIN_DAYS + TEST_DAYS) * RECORDS_PER_DAY
    if len(dataset.timestamps) != expected_records:
        raise ValueError(
            f"expected {expected_records} records for seven full days, "
            f"received {len(dataset.timestamps)}"
        )
    train_stop = TRAIN_DAYS * RECORDS_PER_DAY
    all_prediction_rows: list[dict[str, object]] = []
    per_channel_metric_rows: list[dict[str, object]] = []
    overall_truth: list[bool] = []
    overall_hmm_predictions: list[bool] = []
    overall_persistence_predictions: list[bool] = []
    per_channel_models: dict[str, ChannelOccupancyHMM] = {}

    for channel_index, channel_id in enumerate(CHANNEL_IDS):
        observations = dataset.observations_by_channel[channel_id]
        truth = dataset.ground_truth_by_channel[channel_id]
        if len(observations) != expected_records or len(truth) != expected_records:
            raise ValueError(f"{channel_id} sequence does not cover all seven days")

        training_observations = observations[:train_stop]
        testing_observations = observations[train_stop:]
        testing_truth = truth[train_stop:]
        estimator = ChannelOccupancyHMM(seed=seed + channel_index).fit(
            training_observations
        )
        _, training_posterior = estimator.predict_next_busy_probabilities(
            training_observations
        )
        test_probabilities, _ = estimator.predict_next_busy_probabilities(
            testing_observations,
            previous_posterior=training_posterior,
        )
        per_channel_models[channel_id] = estimator

        # Each observation at test index i predicts ground truth at i+1.
        target_truth = testing_truth[1:]
        hmm_predictions = tuple(value >= 0.5 for value in test_probabilities[:-1])
        persistence_predictions = tuple(
            vote_count >= 2 for vote_count in testing_observations[:-1]
        )
        if len(target_truth) != len(hmm_predictions):
            raise RuntimeError("prediction and target sequence lengths do not match")

        hmm_metrics = classification_metrics(target_truth, hmm_predictions)
        persistence_metrics = classification_metrics(
            target_truth, persistence_predictions
        )
        per_channel_metric_rows.append(_metric_row("HMM", channel_id, hmm_metrics))
        per_channel_metric_rows.append(
            _metric_row("PERSISTENCE", channel_id, persistence_metrics)
        )
        overall_truth.extend(target_truth)
        overall_hmm_predictions.extend(hmm_predictions)
        overall_persistence_predictions.extend(persistence_predictions)

        for index, probability in enumerate(test_probabilities[:-1]):
            based_on_index = train_stop + index
            target_index = based_on_index + 1
            all_prediction_rows.append(
                {
                    "based_on_timestamp": dataset.timestamps[based_on_index].isoformat(),
                    "target_timestamp": dataset.timestamps[target_index].isoformat(),
                    "channel_id": channel_id,
                    "observation_type": "simulated_cooperative_busy_vote_count_0_to_3",
                    "cooperative_busy_vote_count": testing_observations[index],
                    "target_ground_truth": "busy" if testing_truth[index + 1] else "idle",
                    "hmm_busy_probability": f"{probability:.10f}",
                    "hmm_prediction": "busy" if probability >= 0.5 else "idle",
                    "persistence_prediction": (
                        "busy" if testing_observations[index] >= 2 else "idle"
                    ),
                }
            )

    aggregate_hmm = classification_metrics(overall_truth, overall_hmm_predictions)
    aggregate_persistence = classification_metrics(
        overall_truth, overall_persistence_predictions
    )
    per_channel_metric_rows.append(_metric_row("HMM", "OVERALL_MICRO", aggregate_hmm))
    per_channel_metric_rows.append(
        _metric_row("PERSISTENCE", "OVERALL_MICRO", aggregate_persistence)
    )
    summary: dict[str, object] = {
        "training_records_per_channel": train_stop,
        "testing_records_per_channel": RECORDS_PER_DAY,
        "next_record_predictions_per_channel": RECORDS_PER_DAY - 1,
        "total_next_record_predictions": (RECORDS_PER_DAY - 1) * len(CHANNEL_IDS),
        "models": per_channel_models,
    }
    return all_prediction_rows, per_channel_metric_rows, summary


def _write_configuration(
    path: Path,
    *,
    dataset: ObservationDataset,
    seed: int,
    summary: dict[str, object],
) -> None:
    model_names = ["hmmlearn.CategoricalHMM"]
    lines = [
        "SPECTRIX Phase 5 HMM occupancy estimation",
        "Data classification: synthetic ground truth; simulated SU observations.",
        "HMM input: cooperative SU busy-vote count (0..3) only.",
        "Ground truth is used only to generate simulated observations and score held-out predictions; it is not an HMM input.",
        "Train/test split: first six complete days for fitting; seventh day held out chronologically.",
        "Next-record prediction uses observations available through the current record only.",
        "BUSY hidden state: state with the higher fitted expected busy-vote count.",
        "Persistence baseline: next occupancy is predicted from the current cooperative vote count (busy when votes >= 2).",
        f"start_timestamp={dataset.timestamps[0].isoformat()}",
        f"end_timestamp={dataset.timestamps[-1].isoformat()}",
        f"seed={seed}",
        f"hmmlearn_version={hmmlearn.__version__}",
        f"models={','.join(model_names)} x {len(CHANNEL_IDS)} channels",
        f"training_records_per_channel={summary['training_records_per_channel']}",
        f"testing_records_per_channel={summary['testing_records_per_channel']}",
        f"next_record_predictions_per_channel={summary['next_record_predictions_per_channel']}",
        f"total_next_record_predictions={summary['total_next_record_predictions']}",
        f"initialization=sensing-only vote threshold (>=2) with additive smoothing",
        "HMM fitting parameters: n_components=2, n_features=4, n_iter=100, tolerance=0.0001, smoothing=1.0",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and test per-channel HMMs on SPECTRIX simulated sensing data."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("results/channel_occupancy_7days.csv"),
        help="seven-day occupancy database CSV",
    )
    parser.add_argument("--seed", type=int, default=7, help="reproducible sensing/HMM seed")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="destination for new HMM outputs (default: results)",
    )
    args = parser.parse_args()

    dataset = build_observation_dataset(args.database, seed=args.seed)
    prediction_rows, metric_rows, summary = evaluate_dataset(dataset, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _csv_write(
        args.output_dir / PREDICTIONS_FILENAME,
        [
            "based_on_timestamp",
            "target_timestamp",
            "channel_id",
            "observation_type",
            "cooperative_busy_vote_count",
            "target_ground_truth",
            "hmm_busy_probability",
            "hmm_prediction",
            "persistence_prediction",
        ],
        prediction_rows,
    )
    metric_fields = ["method", "channel_id", *[field.name for field in fields(ClassificationMetrics)]]
    _csv_write(args.output_dir / METRICS_FILENAME, metric_fields, metric_rows)
    _write_configuration(
        args.output_dir / CONFIGURATION_FILENAME,
        dataset=dataset,
        seed=args.seed,
        summary=summary,
    )

    print("SPECTRIX Phase 5 HMM estimation complete")
    print(f"Database records: {len(dataset.timestamps):,}")
    print(f"Training records per channel: {summary['training_records_per_channel']:,}")
    print(f"Held-out records per channel: {summary['testing_records_per_channel']:,}")
    print(f"Next-record predictions: {summary['total_next_record_predictions']:,}")
    print(f"Prediction CSV: {(args.output_dir / PREDICTIONS_FILENAME).resolve()}")
    print(f"Metrics CSV: {(args.output_dir / METRICS_FILENAME).resolve()}")
    print(f"Run configuration: {(args.output_dir / CONFIGURATION_FILENAME).resolve()}")
    for row in metric_rows:
        if row["channel_id"] == "OVERALL_MICRO":
            print(
                f"{row['method']} overall: accuracy={row['accuracy']:.4f} "
                f"precision={row['precision']:.4f} recall={row['recall']:.4f} "
                f"F1={row['f1']:.4f}"
            )


if __name__ == "__main__":
    main()
