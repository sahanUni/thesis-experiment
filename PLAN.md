# Final Thesis Experiment Plan

## Status

This project is the final, confirmatory experiment for the master's thesis.
The neighboring projects are development studies used to test hypotheses,
identify failure modes, and establish feasible design choices. Their saved
results may motivate hypotheses, but they are not part of the final comparison.

This document is a living design during development. It becomes the frozen
experiment contract at the final-run gate defined in `TODO.md`.

## Research Question

How do a fixed PID controller, an online PPO gain-scheduled PID controller, and
a direct PPO steering controller trade off nominal tracking accuracy,
robustness, online adaptation, control effort, training cost, and runtime cost
on unseen path-following scenarios?

The study is comparative. It must not be designed to guarantee an RL win. A
valid result may show that different controllers are preferable in different
operating regimes.

### Hypotheses

- **H1 - Nominal accuracy:** A strongly optimized fixed PID will provide the
  best or near-best nominal tracking accuracy with the lowest complexity.
- **H2 - Stationary robustness:** Disturbance-trained PPO controllers will
  degrade less than nominal-only controllers under unseen fixed actuator delay
  and sensor noise.
- **H3 - Online adaptation:** Under an unexpected mid-episode change, the PPO
  gain scheduler and direct PPO will have lower failure-adjusted error and
  faster recovery than a fixed PID.
- **H4 - Cost trade-off:** Any RL robustness advantage will carry measurable
  training, variance, and inference costs that must be reported.

## Experimental Principles

1. All methods run in one controller-neutral simulator and replay the same
   serialized scenarios.
2. The final test manifest remains unopened during development.
3. PID gains, PPO hyperparameters, reward constants, controller code, path
   splits, disturbance settings, metrics, and statistical scripts are frozen
   before the five-seed run.
4. Training reward is not a headline evaluation metric.
5. True Euclidean point-to-path distance is used for scoring. Cross-track error
   alone is not reliable at sharp transitions and self-near paths.
6. Failed and incomplete episodes remain in the analysis.
7. Official results come only from the deterministic batch evaluator. Dashboard
   experiments are demonstrations and never enter the final dataset silently.

## Controller Arms

### Fixed PID benchmarks

The fixed controller uses the same steering PID implementation as the inner PID
of the gain-scheduling agent. The speed PID is identical for every method.

- **PID global nominal:** one gain set optimized over every nominal training
  path and speed.
- **PID global robust:** one gain set optimized over the disturbance-training
  manifest containing delay, noise, and their combination.
- **PID per-path nominal:** one nominal gain set per fixed named benchmark path,
  reported as a privileged path-aware upper-bound benchmark.
- **PID per-path robust:** one disturbance-aware gain set per fixed named
  benchmark path, also reported only as a privileged upper bound.

The global PID is the fair primary fixed baseline because one deployed
controller must handle all paths. Per-path PIDs are stronger but privileged
benchmarks and must be labeled accordingly. They are not tuned for hidden
generated test paths; generated-path generalization compares the global PIDs
and learned controllers only.

### PPO gain-scheduled PID

- PPO acts at 50 Hz and outputs normalized targets for `(K_p, K_i, K_d)`.
- The steering PID runs at 500 Hz.
- Gains are mapped through a frozen calibrated action box and a documented gain
  rate limit.
- PPO controls steering gains only. It does not control target speed.
- Two training regimes are retained:
  - nominal-only training;
  - delay/noise disturbance training.

### Direct PPO steering

- PPO acts at 50 Hz and outputs one normalized steering demand.
- The steering demand is held for ten 500 Hz physics steps.
- It has the same final steering authority and downstream allocator as PID.
- It does not control target speed.
- Two training regimes are retained:
  - nominal-only training;
  - delay/noise disturbance training.

The final learned comparison is therefore a compact 2 x 2 design:

| Controller architecture | Nominal training | Disturbance training |
|---|---:|---:|
| PPO gain-scheduled PID | yes | yes |
| Direct PPO steering | yes | yes |

## Common Control Contract

- MuJoCo physics: 500 Hz (`dt = 0.002 s`).
- Fixed and scheduled steering PID update: 500 Hz.
- PPO decision rate: 50 Hz for both learned architectures.
- Target speeds: `0.3`, `0.5`, and `0.7 m/s`.
- One common 500 Hz speed PID under every steering controller.
- One common steering-priority allocator and wheel-command limits.
- One common actuator delay queue, noise generator, reset pose, corridor rule,
  time limit, termination logic, and trace schema.

### PID implementation

Use one practical PID implementation for both fixed and scheduled PID arms:

- downstream-aware conditional integration for anti-windup;
- a simple first-order exponential low-pass filter on the error derivative;
- filter state initialized from the first measurement to avoid derivative kick;
- one fixed derivative-filter time constant shared by all PID arms;
- no separate unfiltered-PID experimental branch.

The filter time constant is selected with training-only noise probes, recorded,
and frozen before PID optimization. It is not optimized per controller or path.

## Observation Contract

The primary experiment is feedback-only. Neither PPO receives path identity,
future waypoints, curvature preview, hidden delay/noise values, or disturbance
timing.

Both PPO methods receive the same normalized 200 ms causal history of common
measurements:

- measured cross-track error and its filtered/rate representation;
- accumulated cross-track error;
- heading error;
- forward speed and yaw rate;
- target speed;
- previous requested and applied steering quantities;
- wheel utilization, saturation, and allocation-limited flags.

Controller-specific internal state may be appended where it has no common
meaning. The scheduler may observe previous normalized gain actions and PID
integrator state; direct PPO may observe its previous steering action. Every
difference must be listed in the run manifest.

## Path Design

### Design objective

Use a small, interpretable set of literature-supported fixed path patterns for
training, separate paths for model selection, and a locked procedurally
generated set for final geometry generalization. Generated paths are shared
exactly across all methods.

### Proposed fixed training families

The exact parameter values are finalized by path feasibility probes before any
controller training.

1. Constant-curvature arc or circle: steady turning reference.
2. S-curve: smooth curvature reversal.
3. Figure-eight or lemniscate: repeated sign reversal and self-near geometry.
4. Boustrophedon coverage path: straights joined by repeated alternating turns.
5. Continuous-curvature composite path: straight, arc, and smooth curvature
   transition segments.

### Validation and final test paths

- Validation contains fixed paths and generated paths with seeds reserved for
  checkpoint selection and hyperparameter decisions.
- Final testing uses a locked manifest of unseen procedurally generated paths.
- Generated paths are constructed from a smooth bounded curvature profile
  `kappa(s)`, then integrated into heading and XY position.
- Generation bounds include path length, maximum absolute curvature, maximum
  curvature rate, minimum self-separation, and workspace bounds.
- Every path begins at the common reset pose with tangent-aligned heading.
- Paths that violate feasibility or projection assumptions are rejected before
  the test seed list is frozen, using controller-independent geometric checks.
- Deliberately infeasible paths are kept in a separate physical-limit appendix
  and never pooled into the primary result.

### Implemented pre-freeze split

- Training: arc, S-curve, U-turn, slalom, boustrophedon, and one mixed course.
- Validation: sinusoid, quintic lane change, and generated seeds 5000-5001.
- Primary held-out: 20 geometry-screened generated seeds in the 10000 series.
- Structural OOD: figure-eight, hairpin, square, and 60-degree zigzag.
- Physical-limit stress: spiral, sharp zigzags, and the infeasible needle path.

Structural-OOD and physical-limit results are reported separately and are never
pooled into the primary generated-path headline.

The fixed path families are motivated by published use of circular and
figure-eight tracking tests, boustrophedon coverage motion, and
continuous-curvature robot paths. See References.

## PID Optimization

PID tuning is simulation-based robust optimization, not an analytical tuning
rule and not a coarse grid search.

### Method

- Use seeded bounded differential evolution over `(K_p, K_i, K_d)`.
- Evaluate every candidate on the same serialized training scenarios.
- Use fixed search bounds derived from stability and sensitivity probes, not
  from final test performance.
- Cache candidate evaluations and write the full population/convergence history.
- Run deterministic optimizer seeds and record the evaluation budget.
- Select the final candidate with the validation manifest after optimization.
- Freeze gains and calibration bounds before PPO training and final testing.

Differential evolution is appropriate because the simulator objective is
bounded and continuous in its parameters but non-smooth in its outcomes due to
saturation, corridor exits, and completion events. The thesis must state that
the resulting PID is a strong empirically optimized baseline specific to the
declared training distribution.

### Candidate scoring

Optimization uses training-only versions of the final physical metrics. It must
strongly penalize early failure without allowing one arbitrarily weighted score
to hide the components. Each saved candidate row includes completion,
failure-adjusted error, progress, raw tracking error, control variation, and
saturation. Candidate selection is:

1. discard unstable/invalid candidates;
2. retain the highest-completion band;
3. minimize mean failure-adjusted error;
4. break practically negligible ties with lower steering variation and lower
   gain magnitude;
5. confirm the choice on validation scenarios.

## Disturbance Design

Only actuator delay and cross-track sensor noise are in scope. Other plant
disturbances are deferred.

### Evaluation conditions

Each nominal scenario is replayed in four attributable conditions:

1. nominal;
2. actuator delay only;
3. sensor noise only;
4. actuator delay plus sensor noise.

The same four conditions are used in two disturbance modes:

- **Stationary:** the condition is fixed from reset to termination.
- **Transient:** the episode starts at its declared baseline and the condition
  changes once at a fixed simulation time.

One meaningful delay severity and one meaningful noise severity are selected by
training-only plant-sensitivity probes. The values must separate controller
behavior without making every controller fail. Event time, magnitude, and noise
seed are serialized in each scenario. Events occur at simulation time, not at
controller-dependent path progress.

Disturbance-trained agents sample within declared training ranges. Final test
values are fixed before the five-seed run and must not be tuned to favor a
particular controller.

## Training Protocol

### Development phase

- Use one declared development seed.
- Debug only on training and validation manifests.
- Establish environment correctness, reward balance, stable learning,
  checkpoint selection, and a sufficient shared training budget.
- Use learning curves to choose one final PPO decision-step budget that is the
  same for both learned architectures and both training regimes.

### Final phase

- Use five independent, predeclared training seeds for each of the four PPO
  arms, for 20 final training runs.
- Start every final run from scratch with frozen code and configuration.
- Select checkpoints using the same deterministic validation protocol and
  evaluation frequency measured in simulated time.
- Evaluate deterministic policy actions.
- Report every seed. No seed may be dropped because of poor performance unless
  a predeclared infrastructure-failure rule applies and the entire run is
  repeated with the same seed.

### Budget fairness

- Equal 50 Hz PPO decision count and equal simulated seconds for all PPO arms.
- Same policy network class and capacity unless a difference is required by
  action/observation dimensions and documented.
- Same hyperparameter-search budget per learned architecture.
- Reward terms that accumulate over time are scaled per simulated second.
- Record physics ticks, PPO decisions, gradient updates, wall time, CPU/GPU,
  inference latency, and model parameter count.

## Evaluation Metrics

### Primary outcomes

1. **Completion rate:** success within the common corridor and time limit.
2. **Failure-adjusted integrated path error:** includes every episode and pads
   the unused horizon after failure with corridor-boundary error:

   ```text
   J_FA = [ integral_0^t_end distance(t) dt
            + failed * corridor_distance * (T_max - t_end) ] / T_max
   ```

   Completed episodes are padded with zero after completion. Raw components are
   always reported alongside `J_FA`.

### Supporting outcomes

- progress percentage and failure reason;
- mean, RMS, p95, and maximum true path distance;
- IAE and ITAE;
- co-completed paired tracking error, clearly labeled as conditional;
- completion time and achieved forward speed;
- steering-command RMS and total variation per second;
- wheel saturation and allocator-limited fractions;
- pre-event error, peak post-event error, excess post-event IAE/ITAE, and
  sustained recovery time;
- PPO gain trajectories and gain variation for the scheduler;
- training time, sample count, model size, and mean/p99 inference latency.

No metric is hidden when `J_FA` is reported. Reward is used for training
diagnostics only and is never compared across controller architectures.

## Statistical Analysis

- All methods replay identical scenario IDs, noise seeds, and event traces.
- Treat training seed as the unit of algorithm replication; evaluation episodes
  from one trained model are not independent training runs.
- Report per-seed results and aggregate effect sizes with 95% hierarchical
  bootstrap confidence intervals over training seeds and scenarios.
- Use paired completion comparisons on identical scenarios; exact McNemar tests
  may be shown per seed, with consistency summarized across seeds.
- Compare continuous outcomes using paired controller differences, not separate
  unpaired means.
- Report nominal, stationary, and transient tiers separately. Do not pool them
  into one overall winner.
- State practical effect thresholds before opening the final test results.

## Evaluator, Artifacts, and Dashboard

### Official batch evaluator

The command-line evaluator is the only source of official results. It accepts a
frozen scenario manifest and controller artifacts and writes an immutable run
directory containing:

- experiment and scenario manifests;
- model, PID calibration, and source hashes;
- dependency and hardware information;
- one episode-summary row per controller/seed/scenario;
- physics-rate traces;
- aggregate tables, confidence intervals, and figure-ready data;
- validation warnings for missing or non-paired results.

### Interactive dashboard

The dashboard follows the useful interaction pattern of `Blind_PPO`:

- choose controller artifact, path, speed, disturbance condition, magnitude,
  event time, and seed;
- run the selected scenario explicitly;
- overlay reference and controller trajectories;
- compare tracking error, commands, speed, saturation, reward components, and
  PPO gain histories;
- display completion, failure-adjusted error, raw metrics, and recovery metrics;
- save interactive runs outside official final-result directories.

The dashboard may also load immutable batch results for inspection. It must
visibly distinguish official batch artifacts from interactive simulations.

## Supervisor Presentation Story

The final presentation follows one causal narrative:

1. **Problem:** fixed feedback gains may be accurate nominally but cannot change
   when sensing and actuation conditions change.
2. **Fair arena:** establish the common plant, path split, controller information,
   control rates, and failure-aware metrics.
3. **Strong classical baseline:** show how global and path-specific PIDs were
   optimized without test leakage.
4. **Nominal result:** compare accuracy, completion, effort, and cost.
5. **Stationary robustness:** show degradation under fixed delay/noise.
6. **Transient adaptation:** show event response and recovery after unexpected
   changes.
7. **Reliability and cost:** expose seed variance, training cost, and inference
   requirements.
8. **Conclusion:** identify where each architecture is preferable, including a
   PID win if that is what the frozen evidence shows.

Plots should progress from representative trajectories to paired distributions
and seed-level summaries. Best-case videos or dashboard traces illustrate a
mechanism but never replace aggregate evidence.

## Risks and Controls

| Risk | Control |
|---|---|
| Designing for an RL win | Directional hypotheses by regime; strong PID; frozen test set |
| Weak PID baseline | Bounded global optimization plus global/per-path benchmarks |
| Path memorization | Fixed training catalogue plus unseen generated test paths |
| Test leakage | Hidden final manifest and formal freeze gate |
| Noise unfairly breaks PID | Shared derivative filter and anti-windup |
| Early failures look accurate | Completion plus failure-adjusted integrated error |
| Run matrix becomes unmanageable | Four PPO arms, five seeds, one severity per disturbance |
| Different simulator behavior | One plant kernel and serialized paired scenarios |
| Cherry-picked RL checkpoints | One fixed validation selector and fixed evaluation cadence |
| MuJoCo cross-machine variation | Final results generated on one declared machine |

## Deferred Work

- learned speed control;
- curvature-preview or privileged-context policies;
- control-rate ablations;
- mass, friction, actuator-strength, force, and wind disturbances;
- alternative RL algorithms;
- unfiltered PID comparisons;
- deliberately impossible paths in the primary aggregate.

## References

- Storn, R. and Price, K. (1997). Differential Evolution - A Simple and
  Efficient Heuristic for Global Optimization over Continuous Spaces.
  https://doi.org/10.1023/A:1008202821328
- Choset, H. and Pignon, P. (1997). Coverage Path Planning: The Boustrophedon
  Decomposition.
  https://publications.ri.cmu.edu/coverage-path-planning-the-boustrophedon-decomposition
- Shin, D. H., Singh, S., and Whittaker, W. (1992). Path Generation for a Robot
  Vehicle Using Composite Clothoid Segments.
  https://doi.org/10.1016/S1474-6670(17)50946-5
- Palacin, J. et al. (2021). Evaluation of the Path-Tracking Accuracy of a
  Three-Wheeled Omnidirectional Mobile Robot Designed as a Personal Assistant.
  https://doi.org/10.3390/s21217216
