# Final Thesis Experiment Todo List

This checklist is ordered by dependency. Do not begin the final five-seed run
until every item in the freeze gate is complete.

## 0. Project Contract

- [x] Create the `Thesis_Experiment` project directory.
- [x] Write the initial scientific plan.
- [x] Create a separate execution checklist.
- [ ] Review `PLAN.md` with the supervisor.
- [ ] Record supervisor decisions and revise the plan.
- [x] Add a short `README.md` with setup, entry points, and artifact layout.
- [x] Create a decision log for changes made after the initial plan.

**Gate:** research question, controller arms, and final claims are accepted.

## 1. Literature and Path Specification

- [x] Complete a focused literature table for path-following benchmark shapes.
- [x] Document why arc/circle, S-curve, figure-eight, boustrophedon, and
  continuous-curvature composite paths exercise different control behavior.
- [x] Define exact fixed training-path equations and parameters.
- [x] Define validation paths and reserve their seeds.
- [x] Implement the bounded smooth-curvature path generator.
- [x] Define curvature, curvature-rate, length, self-separation, and workspace
  constraints.
- [x] Generate candidate test paths without running any controller on them.
- [x] Validate candidate paths using geometry-only checks.
- [ ] Freeze path IDs and generation seeds in separate training, validation, and
  hidden-test manifests.
- [x] Put infeasible or projection-ambiguous paths in an appendix manifest.

**Gate:** every path is feasible by geometry, starts from the common reset pose,
and belongs to exactly one split.

## 2. Canonical Simulator

- [x] Select the final source versions of the MuJoCo XML, dynamics, allocator,
  PID, path projection, and disturbance machinery.
- [x] Move them behind one controller-neutral simulation kernel; do not retain
  separate PID and direct-RL environment physics implementations.
- [x] Define a controller adapter interface with reset, observation, action,
  trace, and artifact metadata contracts.
- [x] Implement adapters for fixed PID, PPO gain scheduling, and direct PPO.
- [x] Lock 500 Hz physics/PID and 50 Hz PPO timing.
- [x] Verify identical speed PID and steering authority for every controller.
- [x] Define one scenario schema for path, speed, initial state, delay, noise,
  event timing, and RNG seeds.
- [x] Define one common episode-summary and physics-trace schema.
- [x] Add deterministic replay and paired-scenario tests.
- [x] Add tests that delay/noise traces are identical across controller adapters.
- [ ] Add tests for corridor exit, completion, timeout, invalid state, and reset.

**Gate:** the same scripted steering sequence produces the same plant trace
regardless of which controller adapter supplies it.

## 3. Practical PID Implementation

- [x] Retain conditional integration and downstream allocator feedback.
- [x] Add the fixed exponential derivative filter.
- [x] Initialize derivative/filter state without a reset kick.
- [x] Create training-only noise probes for candidate filter time constants.
- [x] Select the simplest filter constant that controls noise without excessive
  phase lag.
- [x] Freeze and document the filter constant before tuning gains.
- [x] Unit-test proportional, integral, derivative, filtering, saturation, and
  anti-windup behavior.
- [x] Verify fixed PID and PPO-scheduled PID instantiate identical PID code.

**Gate:** PID behavior is numerically tested and its noise response is measured.

## 4. Scenario and Disturbance Calibration

- [x] Build nominal paired scenario manifests for all training paths and speeds.
- [x] Measure actuator-delay sensitivity using several PID gains without PPO.
- [x] Measure sensor-noise sensitivity using several PID gains without PPO.
- [x] Choose one delay severity that separates controllers without universal
  failure.
- [x] Choose one noise severity that separates controllers without universal
  failure.
- [x] Choose one transient event time that occurs during all primary paths.
- [x] Define nominal, delay-only, noise-only, and combined conditions.
- [x] Define stationary and transient versions using the same base scenarios.
- [x] Define disturbance-training ranges without reading hidden-test results.
- [ ] Freeze disturbance values, event timing, RNG rules, and manifests.

**Gate:** each disturbance has a measured effect and the combined condition
does not make the benchmark uniformly infeasible.

## 5. PID Optimization

- [x] Establish defensible non-negative bounds for `K_p`, `K_i`, and `K_d`.
- [x] Implement seeded bounded differential-evolution tuning.
- [x] Implement persistent candidate caching and convergence logs.
- [x] Implement invalid-state and early-failure handling in the objective.
- [x] Save completion, failure-adjusted error, progress, tracking, variation,
  saturation, and gain magnitude for every candidate.
- [x] Tune the global nominal PID on nominal training scenarios.
- [x] Tune the global robust PID on the disturbance-training manifest.
- [x] Tune nominal per-path PIDs for fixed named paths as privileged upper bounds.
- [x] Tune robust per-path PIDs for fixed named paths as privileged upper bounds.
- [x] Verify that no per-path PID calibration consumes a hidden generated test
  path or its final scenarios.
- [x] Validate selected gains only on the validation manifest.
- [x] Freeze gain files, action-box derivation, optimizer seeds, bounds, and
  evaluation budgets.
- [x] Produce a supervisor-facing explanation of why differential evolution is
  appropriate for the non-smooth simulator objective.

**Gate:** the global PID is competitive, stable, and demonstrably not selected
from final test performance.

## 6. Common Reward and PPO Training Pipeline

- [x] Define a common task reward from progress, true tracking, physical steering
  variation, completion, and failure.
- [ ] Scale time-accumulating reward terms per simulated second.
- [ ] Audit quitting, circling, crawling, chatter, and saturation incentives.
- [x] Define the common normalized feedback/history observation block.
- [x] Define and document only the necessary controller-specific observation
  fields.
- [x] Implement the normalized scheduler gain action and frozen rate limit.
- [x] Implement the direct steering action with ten-tick zero-order hold.
- [x] Use the same PPO implementation, network class, and decision rate.
- [x] Implement deterministic validation checkpoint selection using physical
  metrics rather than training reward.
- [x] Save complete run configurations, manifests, checkpoints, monitor data,
  training curves, environment versions, and hashes.

**Gate:** smoke runs for both architectures complete and all artifacts can be
loaded by the batch evaluator.

## 7. Single-Seed Development

- [x] Declare the one development seed.
- [ ] Train nominal PPO gain scheduling on training paths.
- [ ] Train disturbance PPO gain scheduling on training paths.
- [ ] Train nominal direct PPO on training paths.
- [ ] Train disturbance direct PPO on training paths.
- [ ] Compare learning stability and validation curves.
- [ ] Diagnose failures using traces, never hidden-test results.
- [ ] Confirm that scheduler gains actually change with observations/events.
- [ ] Confirm that direct PPO does not chatter or exploit termination.
- [ ] Confirm that disturbance-trained agents retain acceptable nominal behavior.
- [ ] Determine one common final PPO decision-step budget from convergence data.
- [ ] Limit hyperparameter iteration to a recorded equal budget per architecture.
- [ ] Run the complete batch-evaluator pipeline on validation manifests.
- [ ] Verify dashboard execution and trace visualizations.

**Gate:** every controller is technically sound, validation results are stable,
and no unresolved bug can plausibly determine the final ranking.

## 8. Metrics and Statistical Pipeline

- [x] Implement completion, progress, and failure-reason metrics.
- [x] Implement failure-adjusted integrated path error exactly as specified.
- [x] Test failure padding for early exit, late exit, timeout, and completion.
- [x] Implement raw mean/RMS/p95/max distance, IAE, and ITAE.
- [x] Implement completion time, speed, control variation, saturation, and
  allocation metrics.
- [x] Implement pre-event, peak, excess-error, and sustained-recovery metrics.
- [x] Implement gain-response metrics for the scheduler.
- [x] Implement training cost, model size, and inference-latency reporting.
- [x] Implement paired episode tables with missing-pair validation.
- [x] Implement hierarchical bootstrap confidence intervals.
- [x] Implement paired continuous effect sizes.
- [ ] Define practical significance thresholds before final evaluation.
- [ ] Generate all final table and figure templates using validation data.

**Gate:** synthetic metric tests pass and the full analysis can be generated
without manual spreadsheet edits.

## 9. Interactive Dashboard

- [x] Build explicit controls for controller artifact, path, speed, disturbance,
  event time, and seed.
- [x] Run simulations only after an explicit user command.
- [x] Overlay trajectories for selected controllers on the same path.
- [x] Plot path error, speed, commands, saturation, and disturbance timing.
- [x] Plot PID gains and gain changes for the scheduling controller.
- [x] Display completion, failure-adjusted error, and all raw metrics.
- [x] Support loading immutable batch-evaluation artifacts.
- [x] Clearly label official batch results versus interactive dashboard runs.
- [x] Store dashboard-generated artifacts outside final-result directories.
- [x] Test scenario replay equivalence between dashboard and batch evaluator.

**Gate:** the dashboard reproduces batch traces for the same scenario and cannot
modify official results.

## 10. Final Freeze

- [ ] Review all unresolved decisions in `PLAN.md` and the decision log.
- [ ] Freeze the canonical plant and controller code commit.
- [x] Freeze PID gains, gain bounds, derivative filter, and allocator settings.
- [ ] Freeze PPO hyperparameters, network, reward, observation, and budget.
- [ ] Freeze training, validation, and hidden-test path manifests.
- [ ] Freeze delay/noise severities, training ranges, event time, and RNG seeds.
- [ ] Freeze the five training seeds.
- [ ] Freeze checkpoint selection, metrics, tests, and statistical scripts.
- [ ] Record dependency lock files and final hardware environment.
- [ ] Run all unit, replay, integration, and smoke tests.
- [ ] Archive the frozen plan and configuration hashes.
- [ ] Obtain supervisor approval to open the final experiment phase.

**Hard gate:** after this point, no outcome-driven code or parameter changes are
allowed. A correctness bug requires documenting the bug, invalidating all
affected final runs, fixing it, and rerunning every affected arm.

## 11. Final Five-Seed Runs

- [ ] Train 5 nominal PPO gain-scheduler seeds.
- [ ] Train 5 disturbance PPO gain-scheduler seeds.
- [ ] Train 5 nominal direct-PPO seeds.
- [ ] Train 5 disturbance direct-PPO seeds.
- [ ] Verify run completeness and artifact hashes without inspecting test scores.
- [ ] Evaluate fixed PID artifacts on the complete final manifest.
- [ ] Evaluate all 20 frozen PPO models on the identical final manifest.
- [ ] Validate that every expected controller/seed/scenario pair exists once.
- [ ] Regenerate any infrastructure-failed run only under the predeclared rule.
- [ ] Seal the immutable raw final-result directory.

**Gate:** complete paired data exist for every declared final scenario.

## 12. Final Analysis and Thesis Story

- [ ] Generate nominal completion, error, effort, and cost tables.
- [ ] Generate stationary delay/noise attribution tables.
- [ ] Generate transient event-response and recovery tables.
- [ ] Generate per-seed plots and hierarchical confidence intervals.
- [ ] Generate paired effect distributions, not only aggregate bars.
- [ ] Inspect and explain both best improvements and worst regressions.
- [ ] Compare nominal-trained and disturbance-trained policies within each
  architecture.
- [ ] Compare global PID, robust global PID, and per-path PID benchmarks.
- [ ] Report training cost, inference latency, and model size.
- [ ] Write conclusions by operating regime rather than forcing one winner.
- [ ] Document rejected hypotheses and negative results.
- [ ] Produce the supervisor presentation in the story order from `PLAN.md`.
- [ ] Export final dashboard examples from preselected representative scenarios
  only after aggregate analysis is complete.
- [ ] Archive scripts, manifests, raw data, processed data, figures, and hashes.

**Done:** the thesis result is reproducible from immutable artifacts, every
claim maps to a predeclared metric, and the complete seed/scenario evidence is
available for review.
