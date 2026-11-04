"""Common queue, retry, transmission, and metric loop for allocation policies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Protocol, Sequence

from simulator import ScenarioConfig, SimulationSnapshot, WirelessSpectrumSimulator
from simulator.metrics import MetricsAccumulator, RunMetrics
from simulator.models import SensingReading, TransmissionResult


@dataclass(slots=True)
class QueuedPacket:
    arrival_slot: int
    failed_attempts: int = 0


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    """The observable allocation view, excluding simulator ground truth."""

    time_slot: int
    packet_arrivals: tuple[int, ...]
    sensing_readings: tuple[SensingReading, ...]
    queue_lengths: tuple[int, ...]
    database_record_index: int | None = None
    database_timestamp: datetime | None = None

    def readings_for_user(self, secondary_user_id: int) -> tuple[SensingReading, ...]:
        readings = tuple(
            reading
            for reading in self.sensing_readings
            if reading.secondary_user_id == secondary_user_id
        )
        if not readings:
            raise ValueError(f"unknown secondary user id: {secondary_user_id}")
        return readings


class AllocationPolicy(Protocol):
    def choose_channels(
        self, snapshot: DecisionSnapshot
    ) -> Sequence[int | None]:
        """Choose channels using the decision-only observation snapshot."""


DecisionHook = Callable[
    [object, DecisionSnapshot, tuple[int | None, ...]], None
]


@dataclass(slots=True)
class _RunContext:
    simulator: WirelessSpectrumSimulator
    scenario: ScenarioConfig
    method: str
    seed: int
    slots: int
    decision_cadence_slots: int
    policy: AllocationPolicy
    metrics: MetricsAccumulator
    queues: list[deque[QueuedPacket]]
    snapshot: SimulationSnapshot
    active_channels: tuple[int | None, ...] | None = None
    on_decision: DecisionHook | None = None


def _make_context(
    simulator: WirelessSpectrumSimulator,
    policy: AllocationPolicy,
    *,
    method: str,
    seed: int,
    slots: int,
    decision_cadence_slots: int,
    on_decision: DecisionHook | None = None,
) -> _RunContext:
    if slots < 1:
        raise ValueError("slots must be at least 1")
    if decision_cadence_slots < 1:
        raise ValueError("decision_cadence_slots must be at least 1")
    scenario = simulator.scenario
    snapshot = simulator.reset(seed=seed)
    return _RunContext(
        simulator=simulator,
        scenario=scenario,
        method=method,
        seed=seed,
        slots=slots,
        decision_cadence_slots=decision_cadence_slots,
        policy=policy,
        metrics=MetricsAccumulator(scenario, method, seed, slots),
        queues=[deque() for _ in range(scenario.num_secondary_users)],
        snapshot=snapshot,
        on_decision=on_decision,
    )


def _decision_snapshot(
    context: _RunContext, snapshot: SimulationSnapshot
) -> DecisionSnapshot:
    source = context.simulator.occupancy_source
    source_record = (
        source.record_for_slot(snapshot.time_slot)
        if source is not None
        else None
    )
    return DecisionSnapshot(
        time_slot=snapshot.time_slot,
        packet_arrivals=snapshot.packet_arrivals,
        sensing_readings=snapshot.sensing_readings,
        queue_lengths=tuple(len(queue) for queue in context.queues),
        database_record_index=(
            source_record.record_index if source_record is not None else None
        ),
        database_timestamp=(
            source_record.timestamp if source_record is not None else None
        ),
    )


def _process_queue_results(
    context: _RunContext,
    snapshot: SimulationSnapshot,
    results: Sequence[TransmissionResult],
) -> tuple[list[float], int]:
    delivered_latencies_ms: list[float] = []
    dropped_this_slot = 0
    for user_id, result in enumerate(results):
        if result.packets_attempted == 0:
            continue

        packet = context.queues[user_id][0]
        if result.outcome == "SUCCESS":
            context.queues[user_id].popleft()
            delay_slots = snapshot.time_slot - packet.arrival_slot + 1
            delivered_latencies_ms.extend(
                [delay_slots * context.scenario.slot_duration_ms]
                * result.packets_delivered
            )
        else:
            packet.failed_attempts += 1
            if packet.failed_attempts >= context.scenario.max_attempts_per_packet:
                context.queues[user_id].popleft()
                dropped_this_slot += 1
    return delivered_latencies_ms, dropped_this_slot


<<<<<<< HEAD
def _advance_one_slot(
    context: _RunContext,
    slot_index: int,
    *,
    advance_environment: bool = True,
) -> None:
=======
def _advance_one_slot(context: _RunContext, slot_index: int) -> None:
>>>>>>> ca209839a0e6c20a8365a3fc13fa2fe94c184397
    snapshot = context.snapshot
    for user_id, arrivals in enumerate(snapshot.packet_arrivals):
        context.queues[user_id].extend(
            QueuedPacket(arrival_slot=snapshot.time_slot)
            for _ in range(arrivals)
        )

    if slot_index % context.decision_cadence_slots == 0:
        decision_snapshot = _decision_snapshot(context, snapshot)
        selected = tuple(context.policy.choose_channels(decision_snapshot))
        if len(selected) != context.scenario.num_secondary_users:
            raise ValueError("allocation policy returned the wrong number of channels")
        context.active_channels = selected
        if context.on_decision is not None:
            context.on_decision(context.policy, decision_snapshot, selected)

    if context.active_channels is None:
        raise RuntimeError("allocation policy did not make an initial decision")
    packets_to_attempt = tuple(1 if queue else 0 for queue in context.queues)
    results = context.simulator.transmit(
        snapshot,
        context.active_channels,
        packets_to_attempt=packets_to_attempt,
    )
    delivered_latencies_ms, dropped_this_slot = _process_queue_results(
        context, snapshot, results
    )
    deferred_this_slot = sum(
        has_packet and channel_id is None
        for has_packet, channel_id in zip(packets_to_attempt, context.active_channels)
    )
    context.metrics.record_slot(
        snapshot,
        results,
        delivered_latencies_ms,
        dropped_packets=dropped_this_slot,
        deferred_packet_slots=deferred_this_slot,
    )

<<<<<<< HEAD
    if advance_environment and slot_index + 1 < context.slots:
=======
    if slot_index + 1 < context.slots:
>>>>>>> ca209839a0e6c20a8365a3fc13fa2fe94c184397
        context.snapshot = context.simulator.step()


def run_policy(
    simulator: WirelessSpectrumSimulator,
    policy: AllocationPolicy,
    *,
    method: str,
    seed: int,
    slots: int,
    decision_cadence_slots: int = 1,
    on_decision: DecisionHook | None = None,
) -> RunMetrics:
    """Run one policy while reusing the shared simulator and metric model."""

    context = _make_context(
        simulator,
        policy,
        method=method,
        seed=seed,
        slots=slots,
        decision_cadence_slots=decision_cadence_slots,
        on_decision=on_decision,
    )
    for slot_index in range(slots):
        _advance_one_slot(context, slot_index)
    return context.metrics.finish(
        pending_packets_at_end=sum(len(queue) for queue in context.queues)
    )


def run_paired_policies(
    standard_simulator: WirelessSpectrumSimulator,
    assm_simulator: WirelessSpectrumSimulator,
    standard_policy: AllocationPolicy,
    assm_policy: AllocationPolicy,
    *,
    seed: int,
    slots: int,
    decision_cadence_slots: int,
    on_assm_decision: DecisionHook | None = None,
) -> tuple[RunMetrics, RunMetrics]:
    """Run Standard and ASSM in lockstep with paired random streams.

<<<<<<< HEAD
    One shared environmental snapshot is generated per simulator slot and
    supplied to both independent policy contexts. This is equivalent to
    advancing two same-seed simulators independently, while avoiding a second
    generation of occupancy, traffic, and sensing data. Each method retains
    its own packet queues, allocation state, and metrics.

    The policy API receives a DecisionSnapshot without ground truth.
=======
    At each decision boundary, the runner verifies that both methods received
    the same primary occupancy, traffic arrivals, and individual sensing
    readings. The policy API receives a DecisionSnapshot without ground truth.
>>>>>>> ca209839a0e6c20a8365a3fc13fa2fe94c184397
    """

    standard = _make_context(
        standard_simulator,
        standard_policy,
        method="STANDARD_10S",
        seed=seed,
        slots=slots,
        decision_cadence_slots=decision_cadence_slots,
    )
    assm = _make_context(
        assm_simulator,
        assm_policy,
        method="ASSM",
        seed=seed,
        slots=slots,
        decision_cadence_slots=decision_cadence_slots,
        on_decision=on_assm_decision,
    )

<<<<<<< HEAD
    if standard_simulator.scenario != assm_simulator.scenario:
        raise ValueError("paired simulators must use identical scenario settings")

    def occupancy_signature(simulator: WirelessSpectrumSimulator) -> tuple[object, ...] | None:
        source = simulator.occupancy_source
        if source is None:
            return None
        return (
            source.csv_path.resolve(),
            source.expected_timestamps,
            source.start_record,
            source.record_count,
            source.steps_per_record,
        )

    if occupancy_signature(standard_simulator) != occupancy_signature(assm_simulator):
        raise ValueError("paired simulators must replay the same occupancy records")
    if standard.snapshot != assm.snapshot:
        raise RuntimeError("paired simulators diverged in their initial environment")

    for slot_index in range(slots):
        shared_snapshot = standard.snapshot
        assm.snapshot = shared_snapshot
        _advance_one_slot(standard, slot_index, advance_environment=False)
        _advance_one_slot(assm, slot_index, advance_environment=False)
        if slot_index + 1 < slots:
            shared_snapshot = standard_simulator.step()
            standard.snapshot = shared_snapshot
            assm.snapshot = shared_snapshot
=======
    for slot_index in range(slots):
        if slot_index % decision_cadence_slots == 0:
            left = standard.snapshot
            right = assm.snapshot
            if (
                left.time_slot != right.time_slot
                or left.primary_occupancy != right.primary_occupancy
                or left.packet_arrivals != right.packet_arrivals
                or left.sensing_readings != right.sensing_readings
            ):
                raise RuntimeError(
                    f"paired environments diverged at decision boundary {slot_index}"
                )
        _advance_one_slot(standard, slot_index)
        _advance_one_slot(assm, slot_index)
>>>>>>> ca209839a0e6c20a8365a3fc13fa2fe94c184397

    standard_metrics = standard.metrics.finish(
        pending_packets_at_end=sum(len(queue) for queue in standard.queues)
    )
    assm_metrics = assm.metrics.finish(
        pending_packets_at_end=sum(len(queue) for queue in assm.queues)
    )
    return standard_metrics, assm_metrics
