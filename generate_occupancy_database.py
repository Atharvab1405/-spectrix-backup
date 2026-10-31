"""Generate and validate SPECTRIX's seven-day synthetic occupancy database."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from simulator.occupancy_config import (
    BROADCAST_ASSUMPTION_NOTE,
    BROADCAST_REFERENCE,
    CATEGORY_ACCESS_NOTE,
    CELLULAR_ASSUMPTION_NOTE,
    DATA_SEMANTICS_NOTE,
    DEFAULT_CONFIG,
    DAYPART_NAMES,
    OccupancyDatabaseConfig,
    daypart_for_hour,
    target_busy_pct,
)


DATABASE_FILENAME = "channel_occupancy_7days.csv"
HOURLY_FILENAME = "hourly_occupancy_summary.csv"
DAYPART_FILENAME = "daypart_occupancy_summary.csv"
DAILY_FILENAME = "daily_occupancy_summary.csv"
REPORT_FILENAME = "database_validation_report.txt"

DATABASE_FIELDS = (
    "timestamp",
    "day_number",
    "weekday",
    "hour",
    "minute",
    "daypart",
    "channel_id",
    "channel_category",
    "occupancy_state",
    "configured_busy_probability_pct",
)


@dataclass(slots=True)
class Aggregate:
    """Counts and configured probability accumulated from CSV rows."""

    total_slots: int = 0
    busy_slots: int = 0
    configured_probability_sum: float = 0.0
    independent_variance_sum: float = 0.0
    category: str = ""

    def add(self, busy: bool, target_pct: float, category: str) -> None:
        probability = target_pct / 100.0
        self.total_slots += 1
        self.busy_slots += int(busy)
        self.configured_probability_sum += probability
        self.independent_variance_sum += probability * (1.0 - probability)
        self.category = category

    @property
    def actual_pct(self) -> float:
        return 100.0 * self.busy_slots / self.total_slots if self.total_slots else 0.0

    @property
    def configured_pct(self) -> float:
        return (
            100.0 * self.configured_probability_sum / self.total_slots
            if self.total_slots
            else 0.0
        )


@dataclass(slots=True)
class AuditResult:
    checks: dict[str, bool]
    details: dict[str, str]
    errors: list[str]
    row_count: int
    timestamp_count: int
    channel_counts: dict[str, int]
    daily_counts: dict[tuple[str, int], int]
    hours_seen: set[int]
    dayparts_seen: set[str]
    hourly: dict[tuple[str, int], Aggregate]
    by_daypart: dict[tuple[str, str], Aggregate]
    daily: dict[tuple[str, int, str, str], Aggregate]
    cellular_overnight: dict[str, Aggregate]
    business_weekend: dict[str, Aggregate]
    state_digest: str
    file_digest: str
    target_group_failures: list[str]


def _start_datetime(start_date: date, timezone_name: str) -> datetime:
    timezone = ZoneInfo(timezone_name)
    return datetime.combine(start_date, time.min).replace(tzinfo=timezone)


def _profile_transition_probability(
    previous_probability: float,
    current_probability: float,
    previous_busy: bool,
    persistence: float,
) -> float:
    """Choose a correlated transition while preserving current marginal rate.

    For a constant target probability, this reduces to a two-state Markov chain
    with the configured lag-one persistence. When the target changes at a
    profile boundary, maximal positive coupling keeps the state continuous while
    preserving the new slot's configured marginal probability.
    """

    previous_probability = min(1.0, max(0.0, previous_probability))
    current_probability = min(1.0, max(0.0, current_probability))
    if previous_probability <= 0.0 or previous_probability >= 1.0:
        return current_probability

    joint_busy = (
        previous_probability * current_probability
        + persistence
        * min(
            previous_probability * (1.0 - current_probability),
            (1.0 - previous_probability) * current_probability,
        )
    )
    if previous_busy:
        conditional = joint_busy / previous_probability
    else:
        conditional = (current_probability - joint_busy) / (1.0 - previous_probability)
    return min(1.0, max(0.0, conditional))


def iter_database_rows(
    config: OccupancyDatabaseConfig,
    seed: int,
    start_date: date,
) -> Iterable[dict[str, object]]:
    """Yield all records chronologically, eight channel rows per timestamp."""

    random_stream = random.Random(seed)
    start = _start_datetime(start_date, config.timezone_name)
    slots_per_day = 24 * 60 * 60 // config.slot_seconds
    total_timestamps = config.days * slots_per_day
    previous_busy: dict[str, bool] = {}
    previous_probability: dict[str, float] = {}

    for slot_index in range(total_timestamps):
        timestamp = start + timedelta(seconds=slot_index * config.slot_seconds)
        day_number = slot_index // slots_per_day + 1
        is_weekend = timestamp.weekday() >= 5
        daypart = daypart_for_hour(timestamp.hour)

        for channel in config.channels:
            target_pct = target_busy_pct(channel, timestamp.hour, is_weekend)
            target_probability = target_pct / 100.0
            if channel.channel_id not in previous_busy:
                busy = random_stream.random() < target_probability
            else:
                conditional_probability = _profile_transition_probability(
                    previous_probability[channel.channel_id],
                    target_probability,
                    previous_busy[channel.channel_id],
                    config.lag_one_persistence,
                )
                busy = random_stream.random() < conditional_probability

            previous_busy[channel.channel_id] = busy
            previous_probability[channel.channel_id] = target_probability
            yield {
                "timestamp": timestamp.isoformat(timespec="seconds"),
                "day_number": day_number,
                "weekday": timestamp.strftime("%A"),
                "hour": timestamp.hour,
                "minute": timestamp.minute,
                "daypart": daypart,
                "channel_id": channel.channel_id,
                "channel_category": channel.category,
                "occupancy_state": "busy" if busy else "idle",
                "configured_busy_probability_pct": f"{target_pct:.2f}",
            }


def write_database(
    path: Path,
    config: OccupancyDatabaseConfig,
    seed: int,
    start_date: date,
) -> None:
    """Write the reproducible database CSV with a stable byte representation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=DATABASE_FIELDS,
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(iter_database_rows(config, seed, start_date))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as data_file:
        for block in iter(lambda: data_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _statistical_tolerance_pct(
    aggregate: Aggregate,
    persistence: float,
    sigma: float,
) -> float:
    """Return a correlation-adjusted, configurable sampling tolerance."""

    if aggregate.total_slots <= 0:
        return 0.0
    autocorrelation_factor = (1.0 + persistence) / (1.0 - persistence)
    standard_error = (
        math.sqrt(aggregate.independent_variance_sum * autocorrelation_factor)
        / aggregate.total_slots
        * 100.0
    )
    return sigma * standard_error


def _boundary_examples_pass() -> bool:
    examples = (
        (time(5, 0, 0), "Morning"),
        (time(11, 59, 50), "Morning"),
        (time(12, 0, 0), "Afternoon"),
        (time(16, 59, 50), "Afternoon"),
        (time(17, 0, 0), "Evening"),
        (time(20, 59, 50), "Evening"),
        (time(21, 0, 0), "Nighttime"),
        (time(23, 59, 50), "Nighttime"),
        (time(0, 0, 0), "Nighttime"),
        (time(4, 59, 50), "Nighttime"),
        (time(5, 0, 0), "Morning"),
    )
    return all(daypart_for_hour(value.hour) == expected for value, expected in examples)


def audit_database(
    path: Path,
    config: OccupancyDatabaseConfig,
    seed: int,
    start_date: date,
) -> AuditResult:
    """Calculate structural and occupancy checks from the generated CSV."""

    channel_ids = [channel.channel_id for channel in config.channels]
    channel_by_id = {channel.channel_id: channel for channel in config.channels}
    slots_per_day = 24 * 60 * 60 // config.slot_seconds
    expected_timestamps = config.days * slots_per_day
    expected_rows = expected_timestamps * len(config.channels)
    start = _start_datetime(start_date, config.timezone_name)

    row_count = 0
    timestamp_count = 0
    sequence_errors = 0
    timestamp_parse_errors = 0
    daypart_errors = 0
    metadata_errors = 0
    state_errors = 0
    target_errors = 0
    channel_counts = {channel_id: 0 for channel_id in channel_ids}
    daily_counts: dict[tuple[str, int], int] = defaultdict(int)
    hours_seen: set[int] = set()
    dayparts_seen: set[str] = set()
    hourly: dict[tuple[str, int], Aggregate] = defaultdict(Aggregate)
    by_daypart: dict[tuple[str, str], Aggregate] = defaultdict(Aggregate)
    daily: dict[tuple[str, int, str, str], Aggregate] = defaultdict(Aggregate)
    cellular_overnight: dict[str, Aggregate] = defaultdict(Aggregate)
    business_weekend: dict[str, Aggregate] = defaultdict(Aggregate)
    state_digest = hashlib.sha256()
    errors: list[str] = []

    def add_error(message: str) -> None:
        if len(errors) < 100:
            errors.append(message)

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if tuple(reader.fieldnames or ()) != DATABASE_FIELDS:
            add_error("CSV header does not match the documented database schema")

        for row in reader:
            row_count += 1
            expected_slot = (row_count - 1) // len(config.channels)
            channel_position = (row_count - 1) % len(config.channels)
            if expected_slot >= expected_timestamps:
                sequence_errors += 1
                continue

            expected_timestamp = start + timedelta(
                seconds=expected_slot * config.slot_seconds
            )
            expected_channel = config.channels[channel_position]
            try:
                timestamp = datetime.fromisoformat(row["timestamp"])
            except (KeyError, TypeError, ValueError):
                timestamp_parse_errors += 1
                add_error(f"invalid timestamp at CSV data row {row_count}")
                continue

            if timestamp != expected_timestamp:
                sequence_errors += 1
                if sequence_errors <= 3:
                    add_error(
                        f"timestamp sequence mismatch at data row {row_count}: "
                        f"expected {expected_timestamp.isoformat()}, got {row['timestamp']}"
                    )
            if row.get("channel_id") != expected_channel.channel_id:
                sequence_errors += 1
                if sequence_errors <= 3:
                    add_error(
                        f"channel order mismatch at data row {row_count}: "
                        f"expected {expected_channel.channel_id}, got {row.get('channel_id')}"
                    )

            channel_id = row.get("channel_id", "")
            channel = channel_by_id.get(channel_id)
            if channel is None:
                metadata_errors += 1
                add_error(f"unknown channel ID at CSV data row {row_count}: {channel_id}")
                continue

            channel_counts[channel_id] += 1
            day_number = expected_slot // slots_per_day + 1
            daily_counts[(channel_id, day_number)] += 1
            hours_seen.add(timestamp.hour)
            computed_daypart = daypart_for_hour(timestamp.hour)
            dayparts_seen.add(computed_daypart)
            if row.get("daypart") != computed_daypart:
                daypart_errors += 1
                if daypart_errors <= 3:
                    add_error(
                        f"wrong daypart at data row {row_count}: "
                        f"expected {computed_daypart}, got {row.get('daypart')}"
                    )
            if (
                row.get("day_number") != str(day_number)
                or row.get("weekday") != timestamp.strftime("%A")
                or row.get("hour") != str(timestamp.hour)
                or row.get("minute") != str(timestamp.minute)
                or row.get("channel_category") != channel.category
            ):
                metadata_errors += 1
                if metadata_errors <= 3:
                    add_error(f"date/channel metadata mismatch at data row {row_count}")

            state = row.get("occupancy_state")
            if state not in ("busy", "idle"):
                state_errors += 1
                add_error(f"invalid occupancy state at data row {row_count}: {state}")
                continue
            busy = state == "busy"
            state_digest.update(b"1" if busy else b"0")
            is_weekend = timestamp.weekday() >= 5
            try:
                target_pct = float(row["configured_busy_probability_pct"])
            except (KeyError, TypeError, ValueError):
                target_errors += 1
                add_error(f"invalid configured probability at data row {row_count}")
                continue

            expected_target = target_busy_pct(channel, timestamp.hour, is_weekend)
            if not math.isclose(target_pct, expected_target, abs_tol=0.005):
                target_errors += 1
                if target_errors <= 3:
                    add_error(
                        f"configured target mismatch for {channel_id} at "
                        f"{timestamp.isoformat()}: expected {expected_target}, got {target_pct}"
                    )

            hourly[(channel_id, timestamp.hour)].add(
                busy, target_pct, channel.category
            )
            by_daypart[(channel_id, computed_daypart)].add(
                busy, target_pct, channel.category
            )
            daily[(channel_id, day_number, timestamp.date().isoformat(), row["weekday"])].add(
                busy, target_pct, channel.category
            )
            if channel.category == "Cellular" and timestamp.hour < 5:
                cellular_overnight[channel_id].add(
                    busy, target_pct, channel.category
                )
            if channel.category == "Business" and is_weekend:
                business_weekend[channel_id].add(
                    busy, target_pct, channel.category
                )

            if channel_position == len(config.channels) - 1:
                timestamp_count += 1

    target_group_failures: list[str] = []
    target_groups_checked = 0
    for group_name, groups in (
        ("hourly", hourly),
        ("daypart", by_daypart),
        ("daily", daily),
    ):
        for key, aggregate in groups.items():
            target_groups_checked += 1
            tolerance = _statistical_tolerance_pct(
                aggregate,
                config.lag_one_persistence,
                config.validation_sigma,
            )
            difference = abs(aggregate.actual_pct - aggregate.configured_pct)
            if difference > tolerance + 1e-9:
                target_group_failures.append(
                    f"{group_name} {key}: actual={aggregate.actual_pct:.3f}% "
                    f"target={aggregate.configured_pct:.3f}% "
                    f"tolerance=±{tolerance:.3f} pp"
                )

    expected_channel_count = expected_timestamps
    channel_count_ok = all(
        channel_counts[channel_id] == expected_channel_count
        for channel_id in channel_ids
    )
    daily_count_ok = all(
        daily_counts[(channel_id, day_number)] == slots_per_day
        for channel_id in channel_ids
        for day_number in range(1, config.days + 1)
    )
    hourly_ok = (
        all((channel_id, hour) in hourly for channel_id in channel_ids for hour in range(24))
        and hours_seen == set(range(24))
    )
    daypart_ok = (
        all((channel_id, daypart) in by_daypart
            for channel_id in channel_ids for daypart in DAYPART_NAMES)
        and dayparts_seen == set(DAYPART_NAMES)
        and _boundary_examples_pass()
    )

    cellular_ok = all(
        aggregate.total_slots > 0
        and aggregate.actual_pct < config.cellular_overnight_max_pct
        for aggregate in cellular_overnight.values()
    ) and len(cellular_overnight) == 2
    business_ok = all(
        aggregate.total_slots > 0
        and aggregate.actual_pct <= config.business_weekend_near_idle_max_pct
        for aggregate in business_weekend.values()
    ) and len(business_weekend) == 2
    evening_gt_night = all(
        by_daypart[(channel.channel_id, "Evening")].actual_pct
        > by_daypart[(channel.channel_id, "Nighttime")].actual_pct
        for channel in config.channels
        if (channel.channel_id, "Evening") in by_daypart
        and (channel.channel_id, "Nighttime") in by_daypart
    )

    checks = {
        "record_count": row_count == expected_rows,
        "per_channel_record_count": channel_count_ok,
        "per_channel_per_day_count": daily_count_ok,
        "seven_day_timestamp_sequence": (
            row_count == expected_rows
            and timestamp_count == expected_timestamps
            and sequence_errors == 0
            and timestamp_parse_errors == 0
        ),
        "no_duplicate_or_missing_channel_timestamps": (
            row_count == expected_rows and sequence_errors == 0
        ),
        "all_24_hours_present": hourly_ok,
        "daypart_coverage_and_boundaries": daypart_ok and daypart_errors == 0,
        "timestamp_and_channel_metadata": metadata_errors == 0,
        "occupancy_state_values": state_errors == 0,
        "configured_probabilities_match_configuration": target_errors == 0,
        "hourly_daypart_daily_targets_within_sampling_tolerance": (
            not target_group_failures and target_groups_checked > 0
        ),
        "cellular_00_to_05_below_25_percent_busy": cellular_ok,
        "business_weekend_near_idle": business_ok,
        "evening_busy_fraction_exceeds_nighttime": evening_gt_night,
    }
    details = {
        "expected_rows": str(expected_rows),
        "expected_timestamps": str(expected_timestamps),
        "target_groups_checked": str(target_groups_checked),
        "target_groups_outside_tolerance": str(len(target_group_failures)),
        "sequence_errors": str(sequence_errors),
        "daypart_errors": str(daypart_errors),
        "metadata_errors": str(metadata_errors),
    }

    return AuditResult(
        checks=checks,
        details=details,
        errors=errors,
        row_count=row_count,
        timestamp_count=timestamp_count,
        channel_counts=channel_counts,
        daily_counts=dict(daily_counts),
        hours_seen=hours_seen,
        dayparts_seen=dayparts_seen,
        hourly=dict(hourly),
        by_daypart=dict(by_daypart),
        daily=dict(daily),
        cellular_overnight=dict(cellular_overnight),
        business_weekend=dict(business_weekend),
        state_digest=state_digest.hexdigest(),
        file_digest=_file_sha256(path),
        target_group_failures=target_group_failures,
    )


def _write_aggregate_summary(
    path: Path,
    groups: dict[tuple[object, ...], Aggregate],
    dimensions: tuple[str, ...],
    channel_by_id: dict[str, object],
) -> None:
    fields = (
        *dimensions,
        "channel_id",
        "channel_category",
        "total_slots",
        "busy_slots",
        "idle_slots",
        "configured_busy_probability_pct",
        "actual_occupancy_pct",
        "absolute_error_pct_points",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key, aggregate in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
            row = dict(zip(dimensions, key[1:]))
            channel_id = str(key[0])
            row.update(
                {
                    "channel_id": channel_id,
                    "channel_category": channel_by_id[channel_id].category,
                    "total_slots": aggregate.total_slots,
                    "busy_slots": aggregate.busy_slots,
                    "idle_slots": aggregate.total_slots - aggregate.busy_slots,
                    "configured_busy_probability_pct": f"{aggregate.configured_pct:.6f}",
                    "actual_occupancy_pct": f"{aggregate.actual_pct:.6f}",
                    "absolute_error_pct_points": (
                        f"{abs(aggregate.actual_pct - aggregate.configured_pct):.6f}"
                    ),
                }
            )
            writer.writerow(row)


def write_summaries(
    output_dir: Path,
    audit: AuditResult,
    config: OccupancyDatabaseConfig,
) -> None:
    channel_by_id = {channel.channel_id: channel for channel in config.channels}
    _write_aggregate_summary(
        output_dir / HOURLY_FILENAME,
        audit.hourly,
        ("hour",),
        channel_by_id,
    )
    _write_aggregate_summary(
        output_dir / DAYPART_FILENAME,
        audit.by_daypart,
        ("daypart",),
        channel_by_id,
    )
    _write_aggregate_summary(
        output_dir / DAILY_FILENAME,
        audit.daily,
        ("day_number", "date", "weekday"),
        channel_by_id,
    )


def _temporary_csv(output_dir: Path, prefix: str) -> Path:
    handle, name = tempfile.mkstemp(
        prefix=prefix,
        suffix=".csv",
        dir=output_dir,
    )
    os.close(handle)
    return Path(name)


def _render_report(
    output_dir: Path,
    config: OccupancyDatabaseConfig,
    seed: int,
    start_date: date,
    audit: AuditResult,
    same_seed_pass: bool,
    different_seed_pass: bool,
    different_seed_behavior_pass: bool,
    alternate_seed: int,
    alternate_audit: AuditResult,
) -> tuple[str, bool]:
    end_exclusive = start_date + timedelta(days=config.days)
    end_exclusive_datetime = _start_datetime(end_exclusive, config.timezone_name)
    last_record_timestamp = end_exclusive_datetime - timedelta(
        seconds=config.slot_seconds
    )
    passed_checks = sum(audit.checks.values())
    overall_pass = (
        passed_checks == len(audit.checks)
        and same_seed_pass
        and different_seed_pass
        and different_seed_behavior_pass
        and all(alternate_audit.checks.values())
    )
    status = "PASS" if overall_pass else "FAIL"
    lines = [
        "SPECTRIX synthetic channel occupancy database validation report",
        "",
        "Data classification: synthetic configurable assumptions, not measured RF data.",
        f"Local time zone: {config.timezone_name}",
        f"Start: {start_date.isoformat()} 00:00:00 (inclusive)",
        f"End:   {end_exclusive.isoformat()} 00:00:00 (exclusive)",
        f"Last record timestamp: {last_record_timestamp.isoformat(timespec='seconds')}",
        f"Time-slot resolution: {config.slot_seconds} seconds",
        f"Default seed: {config.default_seed}; generated seed: {seed}",
        f"Markov lag-one persistence assumption: {config.lag_one_persistence:.2f}",
        f"Broadcast reasoning: {BROADCAST_ASSUMPTION_NOTE}",
        f"Broadcast reference: {BROADCAST_REFERENCE}",
        f"Cellular interpretation: {CELLULAR_ASSUMPTION_NOTE}",
        f"Category access note: {CATEGORY_ACCESS_NOTE}",
        f"Data semantics: {DATA_SEMANTICS_NOTE}",
        "",
        "Database counts calculated from the generated CSV:",
        f"  Data records: {audit.row_count:,}",
        f"  Unique timestamps: {audit.timestamp_count:,}",
        f"  Records per channel: {min(audit.channel_counts.values(), default=0):,} .. "
        f"{max(audit.channel_counts.values(), default=0):,}",
        f"  Records per channel per day: {min(audit.daily_counts.values(), default=0):,} .. "
        f"{max(audit.daily_counts.values(), default=0):,}",
        f"  Hours represented: {len(audit.hours_seen)}",
        f"  Dayparts represented: {', '.join(sorted(audit.dayparts_seen))}",
        "",
        "Occupancy checks calculated from generated records:",
    ]
    lines.append("  Overall busy fraction by channel across all seven days:")
    for channel in config.channels:
        aggregates = [
            aggregate
            for key, aggregate in audit.daily.items()
            if key[0] == channel.channel_id
        ]
        total_slots = sum(aggregate.total_slots for aggregate in aggregates)
        busy_slots = sum(aggregate.busy_slots for aggregate in aggregates)
        configured_sum = sum(
            aggregate.configured_probability_sum for aggregate in aggregates
        )
        actual_pct = 100.0 * busy_slots / total_slots if total_slots else 0.0
        target_pct = 100.0 * configured_sum / total_slots if total_slots else 0.0
        lines.append(
            f"    {channel.channel_id} {channel.category}: "
            f"{actual_pct:.3f}% busy (configured week-average {target_pct:.3f}%)"
        )
    lines.append("  Time-specific checks:")
    for channel_id, aggregate in sorted(audit.cellular_overnight.items()):
        lines.append(
            f"  {channel_id} cellular 00:00-04:59: "
            f"{aggregate.actual_pct:.3f}% busy "
            f"(configured {aggregate.configured_pct:.3f}%)"
        )
    for channel_id, aggregate in sorted(audit.business_weekend.items()):
        lines.append(
            f"  {channel_id} business weekend: "
            f"{aggregate.actual_pct:.3f}% busy "
            f"(configured {aggregate.configured_pct:.3f}%)"
        )
    lines.append("  Evening versus Nighttime by channel:")
    for channel in config.channels:
        evening = audit.by_daypart.get((channel.channel_id, "Evening"))
        nighttime = audit.by_daypart.get((channel.channel_id, "Nighttime"))
        if evening and nighttime:
            lines.append(
                f"    {channel.channel_id} {channel.category}: "
                f"Evening {evening.actual_pct:.3f}%, "
                f"Nighttime {nighttime.actual_pct:.3f}%"
            )
    lines.extend(
        [
            "",
            "Reproducibility calculated from generated files:",
            f"  Same seed {seed} produced byte-identical CSV: "
            f"{'PASS' if same_seed_pass else 'FAIL'}",
            f"  Seed {alternate_seed} produced a different occupancy sequence: "
            f"{'PASS' if different_seed_pass else 'FAIL'}",
            f"  Seed {alternate_seed} passed the same occupancy checks: "
            f"{'PASS' if different_seed_behavior_pass else 'FAIL'}",
            "",
            "Validation checks:",
        ]
    )
    for name, passed in audit.checks.items():
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    lines.extend(
        [
            f"  [{'PASS' if same_seed_pass else 'FAIL'}] same-seed byte-identical replay",
            f"  [{'PASS' if different_seed_pass else 'FAIL'}] different-seed state variation",
            f"  [{'PASS' if different_seed_behavior_pass else 'FAIL'}] alternate-seed behavior",
            f"  [{'PASS' if all(alternate_audit.checks.values()) else 'FAIL'}] alternate-seed audit",
            "",
            f"Target groups checked: {audit.details['target_groups_checked']}",
            f"Groups outside the configured statistical tolerance: "
            f"{audit.details['target_groups_outside_tolerance']}",
            f"Tolerance method: {config.validation_sigma:g} standard errors, "
            "adjusted for configured Markov persistence.",
            "",
            "Database files:",
            f"  {DATABASE_FILENAME}",
            f"  {HOURLY_FILENAME}",
            f"  {DAYPART_FILENAME}",
            f"  {DAILY_FILENAME}",
            "  occupancy_run_configuration.txt",
            "",
            f"Main CSV checks: {passed_checks}/{len(audit.checks)}",
            f"Alternate-seed CSV checks: {sum(alternate_audit.checks.values())}/{len(alternate_audit.checks)}",
            "Same-seed and different-seed checks are listed above.",
            f"Overall validation: {status}",
        ]
    )
    if audit.errors:
        lines.extend(["", "Structural validation details:", *[f"  - {error}" for error in audit.errors]])
    if audit.target_group_failures:
        lines.extend(
            [
                "",
                "Target groups outside tolerance (first 50):",
                *[f"  - {failure}" for failure in audit.target_group_failures[:50]],
            ]
        )
    return "\n".join(lines) + "\n", overall_pass


def _write_config_copy(
    output_dir: Path,
    config: OccupancyDatabaseConfig,
    start_date: date,
    seed: int,
) -> None:
    """Write a run-specific, human-readable configuration with the outputs."""

    path = output_dir / "occupancy_run_configuration.txt"
    lines = [
        "SPECTRIX synthetic occupancy configuration",
        "All probabilities are configurable synthetic assumptions, not measured RF data.",
        f"broadcast_reasoning={BROADCAST_ASSUMPTION_NOTE}",
        f"broadcast_reference={BROADCAST_REFERENCE}",
        f"cellular_interpretation={CELLULAR_ASSUMPTION_NOTE}",
        f"category_access_note={CATEGORY_ACCESS_NOTE}",
        f"data_semantics={DATA_SEMANTICS_NOTE}",
        f"start_date={start_date.isoformat()}",
        f"timezone={config.timezone_name}",
        f"days={config.days}",
        f"slot_seconds={config.slot_seconds}",
        f"seed={seed}",
        f"lag_one_persistence={config.lag_one_persistence}",
        f"validation_sigma={config.validation_sigma}",
        f"cellular_overnight_max_pct={config.cellular_overnight_max_pct}",
        f"business_weekend_near_idle_max_pct={config.business_weekend_near_idle_max_pct}",
        "",
        "Rate window order: morning, afternoon, evening, night_21_23, night_00_04",
        "Daypart labels: Morning=05:00-11:59:50; Afternoon=12:00-16:59:50; "
        "Evening=17:00-20:59:50; Nighttime=21:00-04:59:50 across midnight.",
        "",
    ]
    for channel in config.channels:
        lines.append(
            f"{channel.channel_id} ({channel.category}): "
            f"weekday={','.join(map(str, channel.weekday_busy_pct))}; "
            f"weekend={','.join(map(str, channel.weekend_busy_pct))}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the reproducible seven-day, 10-second SPECTRIX occupancy database."
        )
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_CONFIG.start_date.isoformat(),
        help="local calendar date for day 1 (default: 2026-10-26)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.default_seed,
        help="reproducible random seed (default: 7)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="folder for the database, summaries, and report (default: results)",
    )
    args = parser.parse_args()
    try:
        start_date = date.fromisoformat(args.start_date)
        ZoneInfo(DEFAULT_CONFIG.timezone_name)
    except (TypeError, ValueError, KeyError) as error:
        parser.error(f"invalid start date or time zone: {error}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / DATABASE_FILENAME
    write_database(database_path, DEFAULT_CONFIG, args.seed, start_date)
    audit = audit_database(database_path, DEFAULT_CONFIG, args.seed, start_date)
    write_summaries(output_dir, audit, DEFAULT_CONFIG)
    _write_config_copy(output_dir, DEFAULT_CONFIG, start_date, args.seed)

    same_seed_path = _temporary_csv(output_dir, ".occupancy_same_seed_")
    alternate_seed = args.seed + 1
    alternate_seed_path = _temporary_csv(output_dir, ".occupancy_alt_seed_")
    try:
        write_database(same_seed_path, DEFAULT_CONFIG, args.seed, start_date)
        same_seed_pass = _file_sha256(same_seed_path) == audit.file_digest

        write_database(alternate_seed_path, DEFAULT_CONFIG, alternate_seed, start_date)
        alternate_audit = audit_database(
            alternate_seed_path,
            DEFAULT_CONFIG,
            alternate_seed,
            start_date,
        )
        different_seed_pass = alternate_audit.state_digest != audit.state_digest
        different_seed_behavior_pass = all(
            alternate_audit.checks[name]
            for name in (
                "hourly_daypart_daily_targets_within_sampling_tolerance",
                "cellular_00_to_05_below_25_percent_busy",
                "business_weekend_near_idle",
                "evening_busy_fraction_exceeds_nighttime",
            )
        )
    finally:
        same_seed_path.unlink(missing_ok=True)
        alternate_seed_path.unlink(missing_ok=True)

    report, overall_pass = _render_report(
        output_dir,
        DEFAULT_CONFIG,
        args.seed,
        start_date,
        audit,
        same_seed_pass,
        different_seed_pass,
        different_seed_behavior_pass,
        alternate_seed,
        alternate_audit,
    )
    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(report, encoding="utf-8")

    print("SPECTRIX synthetic occupancy database generated")
    print(f"Data records: {audit.row_count:,}")
    print(f"Unique timestamps: {audit.timestamp_count:,}")
    print(f"Channels: {len(DEFAULT_CONFIG.channels)}")
    last_record_timestamp = _start_datetime(
        start_date + timedelta(days=DEFAULT_CONFIG.days),
        DEFAULT_CONFIG.timezone_name,
    ) - timedelta(seconds=DEFAULT_CONFIG.slot_seconds)
    print(
        f"Date range: {start_date.isoformat()} 00:00:00 through "
        f"{last_record_timestamp.isoformat(timespec='seconds')} (inclusive)"
    )
    print(f"Database: {database_path.resolve()}")
    print(f"Hourly summary: {(output_dir / HOURLY_FILENAME).resolve()}")
    print(f"Daypart summary: {(output_dir / DAYPART_FILENAME).resolve()}")
    print(f"Daily summary: {(output_dir / DAILY_FILENAME).resolve()}")
    print(f"Validation report: {report_path.resolve()}")
    print(f"Validation: {'PASS' if overall_pass else 'FAIL'}")

    if not overall_pass:
        raise SystemExit("occupancy database validation failed; inspect the validation report")


if __name__ == "__main__":
    main()
