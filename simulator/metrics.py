"""Aggregate run metrics for repeatable SPECTRIX experiments."""

from dataclasses import dataclass
from collections.abc import Sequence

from .config import ScenarioConfig
from .models import SimulationSnapshot, TransmissionResult


@dataclass(frozen=True, slots=True)
class RunMetrics:
    method: str
    scenario: str
    seed: int
    slots: int
    generated_packets: int
    attempted_packets: int
    delivered_packets: int
    dropped_packets: int
    pending_packets_at_end: int
    available_channel_slots: int
    utilized_idle_channel_slots: int
    spectrum_utilization_pct: float | None
    throughput_bps: float
    packet_delivery_ratio_pct: float | None
    mean_latency_ms: float | None
    su_collision_packets: int
    primary_user_conflict_packets: int
    low_sinr_failure_packets: int
    deferred_packet_slots: int
    robustness_vs_medium_pdr_pct: float | None = None


class MetricsAccumulator:
    """Collect per-slot events and calculate metrics for one simulation run."""

    def __init__(
        self,
        scenario: ScenarioConfig,
        method: str,
        seed: int,
        slots: int,
    ) -> None:
        if slots < 1:
            raise ValueError("slots must be at least 1")
        self.scenario = scenario
        self.method = method
        self.seed = seed
        self.slots = slots

        self.generated_packets = 0
        self.attempted_packets = 0
        self.delivered_packets = 0
        self.dropped_packets = 0
        self.available_channel_slots = 0
        self.utilized_idle_channel_slots = 0
        self.delivered_bits = 0
        self.latency_total_ms = 0.0
        self.latency_sample_count = 0
        self.su_collision_packets = 0
        self.primary_user_conflict_packets = 0
        self.low_sinr_failure_packets = 0
        self.deferred_packet_slots = 0

    def record_slot(
        self,
        snapshot: SimulationSnapshot,
        results: Sequence[TransmissionResult],
        delivered_packet_latencies_ms: Sequence[float],
        dropped_packets: int = 0,
        deferred_packet_slots: int = 0,
    ) -> None:
        if len(results) != self.scenario.num_secondary_users:
            raise ValueError("transmission results do not match the configured user count")

        self.generated_packets += sum(snapshot.packet_arrivals)
        self.available_channel_slots += sum(
            not is_busy for is_busy in snapshot.primary_occupancy
        )
        self.dropped_packets += dropped_packets
        self.deferred_packet_slots += deferred_packet_slots

        for result in results:
            self.attempted_packets += result.packets_attempted
            if result.outcome == "SUCCESS":
                self.utilized_idle_channel_slots += 1
                self.delivered_packets += result.packets_delivered
                self.delivered_bits += result.throughput_bits
            elif result.outcome == "SU_COLLISION":
                self.su_collision_packets += result.packets_attempted
            elif result.outcome == "PRIMARY_USER_ACTIVE":
                self.primary_user_conflict_packets += result.packets_attempted
            elif result.outcome == "LOW_SINR":
                self.low_sinr_failure_packets += result.packets_attempted

        self.latency_total_ms += sum(delivered_packet_latencies_ms)
        self.latency_sample_count += len(delivered_packet_latencies_ms)

    def finish(self, pending_packets_at_end: int) -> RunMetrics:
        simulation_seconds = (
            self.slots * self.scenario.slot_duration_ms / 1000.0
        )
        utilization_pct = (
            100.0 * self.utilized_idle_channel_slots / self.available_channel_slots
            if self.available_channel_slots
            else None
        )
        pdr_pct = (
            100.0 * self.delivered_packets / self.generated_packets
            if self.generated_packets
            else None
        )
        mean_latency_ms = (
            self.latency_total_ms / self.latency_sample_count
            if self.latency_sample_count
            else None
        )

        return RunMetrics(
            method=self.method,
            scenario=self.scenario.name,
            seed=self.seed,
            slots=self.slots,
            generated_packets=self.generated_packets,
            attempted_packets=self.attempted_packets,
            delivered_packets=self.delivered_packets,
            dropped_packets=self.dropped_packets,
            pending_packets_at_end=pending_packets_at_end,
            available_channel_slots=self.available_channel_slots,
            utilized_idle_channel_slots=self.utilized_idle_channel_slots,
            spectrum_utilization_pct=utilization_pct,
            throughput_bps=self.delivered_bits / simulation_seconds,
            packet_delivery_ratio_pct=pdr_pct,
            mean_latency_ms=mean_latency_ms,
            su_collision_packets=self.su_collision_packets,
            primary_user_conflict_packets=self.primary_user_conflict_packets,
            low_sinr_failure_packets=self.low_sinr_failure_packets,
            deferred_packet_slots=self.deferred_packet_slots,
        )
