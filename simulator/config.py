"""Configuration and named traffic scenarios for the shared simulator."""

from dataclasses import dataclass
import math


SCENARIO_NAMES = ("low", "medium", "heavy", "stress")


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Configurable starting assumptions for one Phase 1 scenario.

    Each channel has one primary-user occupancy process. The traffic presets
    vary secondary-user packet arrivals while keeping the radio assumptions
    common, which makes later comparisons easier to control.
    """

    name: str = "medium"
    num_channels: int = 4
    num_secondary_users: int = 3

    initial_busy_probability: float = 0.25
    idle_to_busy_probability: float = 0.12
    busy_to_idle_probability: float = 0.30

    false_alarm_probability: float = 0.05
    missed_detection_probability: float = 0.10

    traffic_arrival_probability: float = 0.50
    max_packets_per_arrival: int = 2
    spike_interval_slots: int | None = None
    spike_duration_slots: int = 0
    spike_arrival_probability: float = 1.0

    noise_floor_dbm: float = -100.0
    primary_signal_dbm: float = -65.0
    secondary_signal_dbm: float = -72.0
    adjacent_channel_attenuation_db: float = 20.0
    minimum_sinr_db: float = 5.0
    packet_size_bits: int = 12000
    slot_duration_ms: float = 1.0
    max_attempts_per_packet: int = 3

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scenario name must not be empty")
        if self.num_channels < 1:
            raise ValueError("num_channels must be at least 1")
        if self.num_secondary_users < 1:
            raise ValueError("num_secondary_users must be at least 1")
        if self.max_packets_per_arrival < 1:
            raise ValueError("max_packets_per_arrival must be at least 1")

        probabilities = {
            "initial_busy_probability": self.initial_busy_probability,
            "idle_to_busy_probability": self.idle_to_busy_probability,
            "busy_to_idle_probability": self.busy_to_idle_probability,
            "false_alarm_probability": self.false_alarm_probability,
            "missed_detection_probability": self.missed_detection_probability,
            "traffic_arrival_probability": self.traffic_arrival_probability,
            "spike_arrival_probability": self.spike_arrival_probability,
        }
        for label, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be between 0 and 1")

        radio_values = {
            "noise_floor_dbm": self.noise_floor_dbm,
            "primary_signal_dbm": self.primary_signal_dbm,
            "secondary_signal_dbm": self.secondary_signal_dbm,
            "adjacent_channel_attenuation_db": self.adjacent_channel_attenuation_db,
            "minimum_sinr_db": self.minimum_sinr_db,
            "slot_duration_ms": self.slot_duration_ms,
        }
        for label, value in radio_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.adjacent_channel_attenuation_db < 0:
            raise ValueError("adjacent_channel_attenuation_db must not be negative")
        if self.packet_size_bits < 1:
            raise ValueError("packet_size_bits must be at least 1")
        if self.max_attempts_per_packet < 1:
            raise ValueError("max_attempts_per_packet must be at least 1")

        if self.spike_interval_slots is None:
            if self.spike_duration_slots != 0:
                raise ValueError("spike_duration_slots requires spike_interval_slots")
        else:
            if self.spike_interval_slots < 1:
                raise ValueError("spike_interval_slots must be at least 1")
            if not 1 <= self.spike_duration_slots <= self.spike_interval_slots:
                raise ValueError(
                    "spike_duration_slots must be between 1 and spike_interval_slots"
                )

    @classmethod
    def preset(cls, name: str) -> "ScenarioConfig":
        """Create one of the documented traffic presets."""

        normalized_name = name.strip().lower()
        presets = {
            "low": {
                "traffic_arrival_probability": 0.20,
                "max_packets_per_arrival": 1,
            },
            "medium": {
                "traffic_arrival_probability": 0.50,
                "max_packets_per_arrival": 2,
            },
            "heavy": {
                "traffic_arrival_probability": 0.80,
                "max_packets_per_arrival": 3,
            },
            "stress": {
                "traffic_arrival_probability": 0.45,
                "max_packets_per_arrival": 2,
                "spike_interval_slots": 10,
                "spike_duration_slots": 3,
                "spike_arrival_probability": 0.98,
            },
        }
        if normalized_name not in presets:
            choices = ", ".join(SCENARIO_NAMES)
            raise ValueError(f"unknown scenario {name!r}; choose from: {choices}")
        return cls(name=normalized_name, **presets[normalized_name])
