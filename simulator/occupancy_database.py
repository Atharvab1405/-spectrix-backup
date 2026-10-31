"""Validated, streaming playback of the SPECTRIX occupancy database CSV."""

from __future__ import annotations

import csv
from itertools import islice
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import TextIO


CHANNEL_IDS = tuple(f"CH{number:02d}" for number in range(1, 9))
CHANNEL_INDEX = {channel_id: index for index, channel_id in enumerate(CHANNEL_IDS)}
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
SLOT_SECONDS = 10
SIMULATOR_STEP_MS = 1
STEPS_PER_RECORD = SLOT_SECONDS * 1000 // SIMULATOR_STEP_MS
EXPECTED_DAYS = 7
EXPECTED_TIMESTAMPS = EXPECTED_DAYS * 24 * 60 * 60 // SLOT_SECONDS
_VALIDATED_FILE_SIGNATURES: set[tuple[str, int, int, int, int]] = set()


@dataclass(frozen=True, slots=True)
class OccupancyRecord:
    """Ground-truth occupancy for one 10-second database interval."""

    record_index: int
    timestamp: datetime
    channel_states: tuple[bool, ...]


class OccupancyDatabaseReplay:
    """Stream validated database records as simulator channel states.

    A database row group contains eight channels at one timestamp. The group
    remains active for ``steps_per_record`` simulator slots. The production
    default is 10,000 slots, matching one 10-second record at 1 ms per slot.

    ``expected_timestamps`` is configurable to allow small CSV fixtures in
    unit tests; normal project runs should use the seven-day default.
    """

    channel_count = len(CHANNEL_IDS)
    steps_per_record = STEPS_PER_RECORD

    def __init__(
        self,
        csv_path: str | Path,
        *,
        expected_timestamps: int = EXPECTED_TIMESTAMPS,
        start_record: int = 0,
        record_count: int | None = None,
    ) -> None:
        if isinstance(expected_timestamps, bool) or not isinstance(
            expected_timestamps, int
        ):
            raise TypeError("expected_timestamps must be a positive integer")
        if expected_timestamps < 1:
            raise ValueError("expected_timestamps must be at least 1")
        if isinstance(start_record, bool) or not isinstance(start_record, int):
            raise TypeError("start_record must be a non-negative integer")
        if not 0 <= start_record < expected_timestamps:
            raise ValueError("start_record must identify a record in the database")
        if record_count is None:
            record_count = expected_timestamps - start_record
        if isinstance(record_count, bool) or not isinstance(record_count, int):
            raise TypeError("record_count must be a positive integer")
        if record_count < 1 or start_record + record_count > expected_timestamps:
            raise ValueError("record_count must fit within the database from start_record")

        self.csv_path = Path(csv_path)
        self.expected_timestamps = expected_timestamps
        self.start_record = start_record
        self.record_count = record_count
        self.total_simulator_steps = record_count * self.steps_per_record
        self._validate_complete_csv()
        self._file: TextIO | None = None
        self._reader: csv.DictReader | None = None
        self._current_record: OccupancyRecord | None = None
        self._next_record_index = 0
        self.reset()

    def _validate_complete_csv(self) -> None:
        """Validate schema, channel mapping, full coverage, and time ordering."""

        try:
            file_stat = self.csv_path.stat()
            signature = (
                str(self.csv_path.resolve()),
                file_stat.st_size,
                file_stat.st_mtime_ns,
                file_stat.st_ctime_ns,
                self.expected_timestamps,
            )
            if signature in _VALIDATED_FILE_SIGNATURES:
                return
            with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                if tuple(reader.fieldnames or ()) != DATABASE_FIELDS:
                    raise ValueError("occupancy CSV header does not match the expected schema")

                first_timestamp: datetime | None = None
                expected_timestamp: datetime | None = None
                for record_index in range(self.expected_timestamps):
                    rows = [next(reader, None) for _ in CHANNEL_IDS]
                    if any(row is None for row in rows):
                        raise ValueError(
                            f"occupancy CSV ended before timestamp group {record_index}"
                        )

                    parsed_timestamps: list[datetime] = []
                    states: list[bool] = []
                    for channel_index, row in enumerate(rows):
                        assert row is not None
                        channel_id = row.get("channel_id")
                        if channel_id != CHANNEL_IDS[channel_index]:
                            raise ValueError(
                                f"timestamp group {record_index} must list channels "
                                f"{', '.join(CHANNEL_IDS)} in order; found {channel_id!r}"
                            )
                        try:
                            timestamp = datetime.fromisoformat(row["timestamp"])
                        except (KeyError, TypeError, ValueError) as error:
                            raise ValueError(
                                f"invalid timestamp at record {record_index}: {error}"
                            ) from error
                        if timestamp.tzinfo is None:
                            raise ValueError(
                                f"timestamp at record {record_index} must include a time zone"
                            )
                        parsed_timestamps.append(timestamp)

                        state = row.get("occupancy_state", "").strip().lower()
                        if state not in {"busy", "idle"}:
                            raise ValueError(
                                f"invalid occupancy_state at record {record_index}: {state!r}"
                            )
                        states.append(state == "busy")

                    group_timestamp = parsed_timestamps[0]
                    if any(value != group_timestamp for value in parsed_timestamps[1:]):
                        raise ValueError(
                            f"channels in timestamp group {record_index} have different timestamps"
                        )

                    if first_timestamp is None:
                        first_timestamp = group_timestamp
                        if first_timestamp.timetz().replace(tzinfo=None) != time.min:
                            raise ValueError("the first database timestamp must be local midnight")
                        expected_timestamp = first_timestamp

                    if group_timestamp != expected_timestamp:
                        raise ValueError(
                            f"timestamp group {record_index} is out of order or not "
                            f"{SLOT_SECONDS}-second resolution; expected {expected_timestamp}, "
                            f"found {group_timestamp}"
                        )
                    expected_timestamp = group_timestamp + timedelta(seconds=SLOT_SECONDS)

                if next(reader, None) is not None:
                    raise ValueError(
                        f"occupancy CSV contains more than {self.expected_timestamps} timestamp groups"
                    )
            final_stat = self.csv_path.stat()
            final_signature = (
                str(self.csv_path.resolve()),
                final_stat.st_size,
                final_stat.st_mtime_ns,
                final_stat.st_ctime_ns,
                self.expected_timestamps,
            )
            if final_signature != signature:
                raise OSError("occupancy CSV changed while it was being validated")
            _VALIDATED_FILE_SIGNATURES.add(signature)
        except OSError as error:
            raise OSError(f"cannot read occupancy database {self.csv_path}: {error}") from error

    def reset(self) -> None:
        """Restart playback at the configured chronological record offset."""

        if self._file is not None:
            self._file.close()
        self._file = self.csv_path.open("r", newline="", encoding="utf-8")
        header = self._file.readline().rstrip("\r\n")
        if tuple(header.split(",")) != DATABASE_FIELDS:
            raise RuntimeError("occupancy CSV header changed after validation")
        lines_to_skip = self.start_record * len(CHANNEL_IDS)
        skipped = sum(1 for _ in islice(self._file, lines_to_skip))
        if skipped != lines_to_skip:
            raise RuntimeError("occupancy CSV ended while seeking to start_record")
        self._reader = csv.DictReader(self._file, fieldnames=DATABASE_FIELDS)
        self._current_record = None
        self._next_record_index = self.start_record

    def close(self) -> None:
        """Close the underlying CSV stream."""

        if self._file is not None:
            self._file.close()
            self._file = None
            self._reader = None

    def _read_next_record(self) -> OccupancyRecord:
        if self._reader is None:
            raise RuntimeError("playback is closed; call reset() before reading")
        record_index = self._next_record_index
        rows = [next(self._reader, None) for _ in CHANNEL_IDS]
        if any(row is None for row in rows):
            raise RuntimeError(
                f"occupancy CSV changed after validation near record {record_index}"
            )

        first_row = rows[0]
        assert first_row is not None
        timestamp = datetime.fromisoformat(first_row["timestamp"])
        states: list[bool] = []
        for channel_index, row in enumerate(rows):
            assert row is not None
            if row["channel_id"] != CHANNEL_IDS[channel_index]:
                raise RuntimeError(
                    f"occupancy CSV channel order changed near record {record_index}"
                )
            if datetime.fromisoformat(row["timestamp"]) != timestamp:
                raise RuntimeError(
                    f"occupancy CSV timestamp changed within group {record_index}"
                )
            states.append(row["occupancy_state"].strip().lower() == "busy")

        record = OccupancyRecord(
            record_index=record_index,
            timestamp=timestamp,
            channel_states=tuple(states),
        )
        self._next_record_index += 1
        return record

    def record_for_slot(self, time_slot: int) -> OccupancyRecord:
        """Return the database record active at a zero-based 1 ms slot.

        The final valid slot is ``total_simulator_steps - 1``. Asking for the
        next slot raises ``StopIteration``; playback never wraps automatically.
        """

        if isinstance(time_slot, bool) or not isinstance(time_slot, int):
            raise TypeError("time_slot must be a non-negative integer")
        if time_slot < 0:
            raise ValueError("time_slot must not be negative")
        if time_slot >= self.total_simulator_steps:
            raise StopIteration("occupancy database playback has reached its end")

        target_index = self.start_record + time_slot // self.steps_per_record
        if target_index >= self.start_record + self.record_count:
            raise StopIteration("occupancy database playback has reached its end")
        if (
            self._current_record is not None
            and target_index < self._current_record.record_index
        ):
            raise ValueError("playback cannot move backward; call reset() first")

        while self._next_record_index <= target_index:
            self._current_record = self._read_next_record()
        assert self._current_record is not None
        return self._current_record

    def state_for_slot(self, time_slot: int) -> tuple[bool, ...]:
        """Return the eight Boolean channel states active at ``time_slot``."""

        return self.record_for_slot(time_slot).channel_states

    def timestamp_for_slot(self, time_slot: int) -> datetime:
        """Return the source database timestamp active at ``time_slot``."""

        return self.record_for_slot(time_slot).timestamp

    def __enter__(self) -> "OccupancyDatabaseReplay":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
