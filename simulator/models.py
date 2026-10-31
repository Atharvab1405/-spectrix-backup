"""Immutable data returned by the Phase 1 simulator."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SensingReading:
    """One secondary user's simulated reading for one channel."""

    secondary_user_id: int
    channel_id: int
    sensed_busy: bool
    rssi_dbm: float
    snr_db: float
    sinr_db: float
    interference_dbm: float | None


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
    """State and observations for a single zero-based time slot.

    ``primary_occupancy`` is simulator ground truth for diagnostics and later
    metric calculations. Decision policies should use ``sensing_readings``
    instead of reading this field.
    """

    time_slot: int
    primary_occupancy: tuple[bool, ...]
    packet_arrivals: tuple[int, ...]
    sensing_readings: tuple[SensingReading, ...]
    queue_lengths: tuple[int, ...] = ()

    def readings_for_user(self, secondary_user_id: int) -> tuple[SensingReading, ...]:
        """Return channel readings in channel-id order for one secondary user."""

        if (
            isinstance(secondary_user_id, int)
            and not isinstance(secondary_user_id, bool)
            and secondary_user_id >= 0
        ):
            channel_count = len(self.primary_occupancy)
            start = secondary_user_id * channel_count
            contiguous = self.sensing_readings[start : start + channel_count]
            if len(contiguous) == channel_count and all(
                reading.secondary_user_id == secondary_user_id
                and reading.channel_id == channel_id
                for channel_id, reading in enumerate(contiguous)
            ):
                return contiguous

        readings = tuple(
            reading
            for reading in self.sensing_readings
            if reading.secondary_user_id == secondary_user_id
        )
        if not readings:
            raise ValueError(f"unknown secondary user id: {secondary_user_id}")
        return readings


@dataclass(frozen=True, slots=True)
class TransmissionResult:
    """Outcome for one secondary user's offered traffic in a single slot."""

    secondary_user_id: int
    channel_id: int | None
    packets_generated: int
    packets_attempted: int
    packets_delivered: int
    outcome: str
    throughput_bits: int
