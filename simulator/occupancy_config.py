"""Configurable assumptions for the synthetic SPECTRIX occupancy database.

All occupancy targets in this module are synthetic model inputs. They are not
measurements from real radio equipment or universal duty-cycle values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


RATE_WINDOWS = (
    "morning",       # 05:00 through 11:59:50
    "afternoon",     # 12:00 through 16:59:50
    "evening",       # 17:00 through 20:59:50
    "night_21_23",   # 21:00 through 23:59:50; daypart label Nighttime
    "night_00_04",   # 00:00 through 04:59:50; daypart label Nighttime
)

DAYPART_NAMES = ("Morning", "Afternoon", "Evening", "Nighttime")

BROADCAST_ASSUMPTION_NOTE = (
    "Broadcast targets are kept high (96-99%) with only small daypart changes to "
    "represent near-continuous transmission with modest modeled variation. "
    "They are configurable synthetic assumptions, not universal measured values."
)
BROADCAST_REFERENCE = (
    "https://cseweb.ucsd.edu/~nibhaska/papers/spectrumusage.pdf"
)
CELLULAR_ASSUMPTION_NOTE = (
    "Cellular busy states represent synthetic traffic-level channel activity, "
    "not universal cellular RF presence or continuous downlink energy."
)
DATA_SEMANTICS_NOTE = (
    "Each row is ground-truth synthetic occupancy for the 10-second interval "
    "beginning at timestamp. Busy maps to True and idle maps to False in the "
    "shared simulator's primary_occupancy channel order. No sensing, RSSI/SNR, "
    "or predicted values are included."
)
CATEGORY_ACCESS_NOTE = (
    "The PS2 written requirements identify Cellular and Business as licensed, "
    "and ISM as unlicensed. They do not specify a regional frequency plan."
)


@dataclass(frozen=True, slots=True)
class ChannelProfile:
    """One channel's target busy probabilities, expressed as percentages."""

    channel_id: str
    category: str
    weekday_busy_pct: tuple[float, ...]
    weekend_busy_pct: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.weekday_busy_pct) != len(RATE_WINDOWS):
            raise ValueError(f"{self.channel_id}: weekday profile needs five values")
        if len(self.weekend_busy_pct) != len(RATE_WINDOWS):
            raise ValueError(f"{self.channel_id}: weekend profile needs five values")
        for value in (*self.weekday_busy_pct, *self.weekend_busy_pct):
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{self.channel_id}: busy percentages must be 0..100")


@dataclass(frozen=True, slots=True)
class OccupancyDatabaseConfig:
    """Settings for one reproducible, seven-day synthetic database."""

    start_date: date = date(2026, 10, 26)
    timezone_name: str = "Asia/Kolkata"
    days: int = 7
    slot_seconds: int = 10
    default_seed: int = 7

    # The existing simulator's default Markov chain has lag-one persistence
    # 1 - 0.12 - 0.30 = 0.58. Reuse that temporal correlation as a configurable
    # synthetic assumption while calibrating each channel to its target rate.
    lag_one_persistence: float = 0.58

    # Approximate validation band: four standard errors after adjusting for
    # Markov autocorrelation. This is a configurable synthetic-data check.
    validation_sigma: float = 4.0
    cellular_overnight_max_pct: float = 25.0
    business_weekend_near_idle_max_pct: float = 5.0

    channels: tuple[ChannelProfile, ...] = (
        ChannelProfile(
            "CH01", "Broadcast",
            (98, 98, 99, 98, 97),
            (98, 98, 99, 98, 97),
        ),
        ChannelProfile(
            "CH02", "Broadcast",
            (96, 97, 98, 97, 96),
            (96, 97, 98, 97, 96),
        ),
        ChannelProfile(
            "CH03", "Cellular",
            (40, 55, 60, 45, 20),
            (40, 55, 60, 45, 20),
        ),
        ChannelProfile(
            "CH04", "Cellular",
            (35, 50, 55, 40, 18),
            (35, 50, 55, 40, 18),
        ),
        ChannelProfile(
            "CH05", "Business",
            (30, 60, 25, 5, 2),
            (1, 1, 1, 1, 1),
        ),
        ChannelProfile(
            "CH06", "Business",
            (25, 55, 20, 4, 1),
            (1, 1, 1, 1, 1),
        ),
        ChannelProfile(
            "CH07", "ISM",
            (15, 25, 45, 30, 15),
            (20, 35, 55, 40, 20),
        ),
        ChannelProfile(
            "CH08", "ISM",
            (12, 22, 40, 27, 12),
            (17, 30, 50, 37, 17),
        ),
    )

    def __post_init__(self) -> None:
        if self.days != 7:
            raise ValueError("the occupancy database must cover exactly seven days")
        if self.slot_seconds != 10:
            raise ValueError("the occupancy database resolution must be 10 seconds")
        if len(self.channels) != 8:
            raise ValueError("exactly eight channel profiles are required")
        if len({channel.channel_id for channel in self.channels}) != 8:
            raise ValueError("channel IDs must be unique")
        expected_categories = {
            "Broadcast": 2,
            "Cellular": 2,
            "Business": 2,
            "ISM": 2,
        }
        actual_categories = {
            category: sum(ch.category == category for ch in self.channels)
            for category in expected_categories
        }
        if actual_categories != expected_categories:
            raise ValueError("there must be exactly two channels in each category")
        if not 0.0 <= self.lag_one_persistence < 1.0:
            raise ValueError("lag_one_persistence must be in [0, 1)")
        if self.validation_sigma <= 0.0:
            raise ValueError("validation_sigma must be positive")
        if not 0.0 < self.cellular_overnight_max_pct <= 100.0:
            raise ValueError("cellular_overnight_max_pct must be in (0, 100]")
        if not 0.0 < self.business_weekend_near_idle_max_pct <= 100.0:
            raise ValueError("business weekend limit must be in (0, 100]")


DEFAULT_CONFIG = OccupancyDatabaseConfig()


def rate_window_for_hour(hour: int) -> str:
    """Return the configurable-profile window for an hour of local time."""

    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    if 21 <= hour < 24:
        return "night_21_23"
    if 0 <= hour < 5:
        return "night_00_04"
    raise ValueError(f"hour must be between 0 and 23, got {hour}")


def daypart_for_hour(hour: int) -> str:
    """Map local hours to the four required labels."""

    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    if 21 <= hour < 24 or 0 <= hour < 5:
        return "Nighttime"
    raise ValueError(f"hour must be between 0 and 23, got {hour}")


def target_busy_pct(
    channel: ChannelProfile,
    hour: int,
    is_weekend: bool,
) -> float:
    """Return the configured target for a channel and local timestamp."""

    window = rate_window_for_hour(hour)
    index = RATE_WINDOWS.index(window)
    profile = channel.weekend_busy_pct if is_weekend else channel.weekday_busy_pct
    return profile[index]
