# SPECTRIX

**SPECTRIX — AI-Based Cognitive Radio Spectrum Management for Dynamic Wireless Networks** is a Python simulation project. Part A builds and evaluates an Adaptive Spectrum Sensing and Management (ASSM) policy against a conventional Standard baseline in one shared wireless simulator.

All occupancy and sensing data in this project are **synthetic simulation data**. Results are not measurements of real-world radio occupancy.

## Project status

- The shared simulator and Standard baseline are implemented.
- The configurable seven-day, eight-channel occupancy database and validation outputs are present.
- Part A components are integrated: cooperative sensing, a two-state HMM, fuzzy channel suitability, and ASSM allocation.
- The HMM day-seven evaluation has been run and saved.
- Phase 7 regression tests passed (39 tests in the reported run), and the Standard/ASSM pipeline completed a small smoke run.
- **The full Phase 7 Standard-versus-ASSM evaluation has not yet been run.** The smoke run is a pipeline check, not a complete performance study.
- Part B (LSTM/deep reinforcement learning) and the interactive dashboard are not implemented in the current project snapshot.

## Part A: ASSM workflow

1. A seeded occupancy generator creates busy/idle ground truth for eight channels at 10-second resolution over seven consecutive days. Occupancy probabilities and start date are configurable.
2. Simulated secondary users (SUs) observe occupancy probabilistically and report busy or idle. This is a probabilistic observation model, **not a threshold-based energy detector**.
3. Cooperative sensing fuses SU reports with configurable K-of-N voting; strict majority is the default.
4. A two-state categorical HMM is trained per channel on cooperative busy-vote counts from days 1–6 and evaluated chronologically on day 7. ASSM uses the current-record filtered busy probability, based only on observations available at that decision.
5. Fuzzy logic maps occupancy probability, SNR, and interference to a configurable 0–100 channel-suitability score, with fired-rule explanations.
6. ASSM rejects cooperatively busy channels and channels whose estimated busy probability crosses configured limits. It applies a fuzzy suitability threshold to remaining candidates, then allocates distinct eligible channels to SUs.
7. Standard and ASSM are compared using the same simulator, occupancy replay, traffic/sensing streams, scenario, seed, record window, and decision cadence. Each 10-second database record is held for 10,000 simulator steps at 1 ms per step.

Database ground truth is used to generate simulated sensing and evaluate completed predictions and transmissions. Ground truth is not an input to ASSM's allocation decision.

## Channels, scenarios, and measurements

The occupancy database has eight channels: two each in Broadcast, Cellular, Business, and ISM categories. It contains **483,840 channel-slot records** (8 channels × 8,640 ten-second slots/day × 7 days).

The shared simulator includes low, medium, heavy, and stress traffic scenarios. It measures:

- **Spectrum utilization:** successful use of primary-idle channel slots, following the simulator's metric definition.
- **Throughput:** delivered payload bits per simulated second.
- **Packet delivery ratio (PDR):** delivered packets divided by generated packets.
- **Latency:** average time from packet arrival to successful delivery.
- **SU collisions:** overlapping secondary-user channel attempts.
- **Primary-user conflicts:** SU attempts on channels occupied by primary users.
- **Robustness:** stress-scenario PDR relative to medium-scenario PDR for the same seed.

The simulator includes packet queues and retries. Occupancy and allocation thresholds are configurable synthetic engineering assumptions, not proven-optimal values.

## Current measured results

### HMM evaluation

The saved seed-7 HMM run trained on the first six days and evaluated the full seventh day (8,640 records per channel). The recorded aggregate metrics were:

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| HMM | 0.9006 | 0.8826 | 0.8811 | 0.8818 |
| Persistence baseline | 0.8943 | 0.8813 | 0.8654 | 0.8733 |

These are measured results from the synthetic simulation dataset. See `results/hmm_metrics.csv` and `results/hmm_occupancy_predictions.csv`.

### Phase 7 smoke run

The saved smoke run used **two held-out records**, the medium scenario, one repetition, and seed 7. It verifies that the paired Standard/ASSM workflow runs and produces artifacts. Its performance values are preliminary and should not be treated as final conclusions. The smoke run did not include the stress scenario, so its robustness table has no data rows.

Smoke outputs are under `results/phase7_smoke_desktop/`. The run estimated approximately **162.53 hours (6.77 days)** for the configured full evaluation by scaling from measured runtime. This is a runtime estimate, not a completed run or guarantee.

## Output files

The generator and experiments write actual outputs under `results/`:

- `channel_occupancy_7days.csv`: full synthetic channel-slot ground truth.
- `hourly_occupancy_summary.csv`, `daypart_occupancy_summary.csv`, `daily_occupancy_summary.csv`: occupancy summaries.
- `database_validation_report.txt`: calculated database validation results.
- `hmm_occupancy_predictions.csv`, `hmm_metrics.csv`, `hmm_run_configuration.txt`: HMM outputs and run settings.
- `phase7_smoke_desktop/`: smoke-run CSVs, ASSM decision and HMM logs, comparison tables, PNG graphs, report, configuration, and timing files.

A full Phase 7 run writes to a separate output directory, for example `results/phase7/`.

## Setup

Use Python 3.10 or newer. The HMM and graph scripts require NumPy, hmmlearn, and Matplotlib. From the project folder, install them with:

```powershell
python -m pip install numpy hmmlearn matplotlib
```

## Commands

Run the short Standard simulator demonstration:

```powershell
python -B main.py --scenario medium --steps 5 --seed 7
```

Run the HMM evaluation:

```powershell
python -B -m experiments.run_hmm_estimation --seed 7
```

Run the regression tests:

```powershell
python -B -m unittest discover -s tests -p "test_*.py" -v
```

Run the small Phase 7 smoke check:

```powershell
python -B -m experiments.run_assm_comparison --scenarios medium --records 2 --repetitions 1 --seed 7 --output-dir results/phase7_smoke_desktop
```

The following command runs the much larger paired evaluation across all scenarios and repetitions. It was **not** run when this README was updated; the runtime estimate above suggests it may take several days:

```powershell
python -B -m experiments.run_assm_comparison --scenarios low medium heavy stress --records 8640 --repetitions 3 --seed 7 --output-dir results/phase7
```

## Interpreting results

Use CSVs as the numerical source of truth and PNG graphs as visual summaries. Compare Standard and ASSM across all six metrics, scenarios, and repeated seeds. Higher utilization or throughput alone does not establish a better policy: consider PDR, latency, collisions, primary-user conflicts, and robustness together. Report the full evaluation separately from the smoke run, and state that the occupancy model is synthetic.

## Tests and reproducibility

The test suite covers cooperative fusion, HMM estimation, fuzzy decisions, occupancy replay, ASSM allocation, paired comparison behavior, and regressions. Runs accept configurable seeds; seed 7 is the default used in project examples. The recorded 39-test pass and smoke outputs establish software checks and pipeline execution. They do not replace the pending full Phase 7 evaluation.
