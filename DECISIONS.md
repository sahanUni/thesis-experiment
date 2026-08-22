# Experiment Decision Log

This log records protocol changes after the initial plan. Parameter changes made
before the final freeze are allowed only with a reason and validation evidence.
After the freeze, correctness fixes invalidate every affected final run.

## 2026-08-19: Initial implementation

- Use one `PathFollowingEnv` for fixed PID, PPO gain scheduling, and direct PPO.
- Keep MuJoCo physics and PID updates at 500 Hz; both PPO policies act at 50 Hz.
- Restrict disturbances to actuator delay and cross-track sensor noise.
- Use a fixed training catalogue, fixed plus seeded generated validation paths,
  and separate seeded generated held-out paths.
- Tune PID gains with bounded, seeded differential evolution and confirm the
  training shortlist on validation scenarios. SIMC is not used because the
  closed-loop path-following plant does not provide the simple process model
  and step-response assumptions that make SIMC attractive.
- Use completion and failure-adjusted integrated path error as co-primary
  outcomes while retaining every raw tracking, effort, and failure metric.
- Keep the dashboard read-only over batch artifacts. It cannot write into or
  recompute an official result directory.
- PID candidates use a declared balanced subset of the training manifest:
  nominal scenarios at all speeds, all stationary conditions at 0.5 m/s, and
  the transient combined condition at 0.5 m/s. This preserves every training
  geometry and disturbance type while keeping differential evolution
  computationally feasible. Final evaluation still uses complete manifests.

## 2026-08-19: Runaway guard correction

The first integration smoke test exposed a 2 m absolute-coordinate guard that
terminated valid paths. The guard was restored to the inherited 20 m numerical
safety bound. The independent 1 m path corridor remains the tracking-failure
criterion. No training or final experiment had been run.

## 2026-08-21: Derivative-filter selection

- Set the derivative-filter time constant to `0.01 s` before PID calibration.
- Noise-only probes showed that filtering is necessary to prevent command
  chatter and saturation. Combined delay/noise checks showed that larger time
  constants add excessive phase lag when used with the selected `0.06 s`
  actuator delay.
- The selected `0.01 s` value suppressed the unfiltered noise response while
  preserving better combined-condition tracking than `0.02 s`, `0.05 s`, or
  `0.10 s`.

## 2026-08-21: Practical PID selection and Kp boundary audit

- Treat candidates within `0.5%` of the best failure-adjusted error in the
  highest-completion band as practically equivalent. Select the lowest steering
  variation, then the lowest gain magnitude, inside that band.
- Validate five distinct error/smoothness/gain trade-offs rather than numerical
  optimizer-polishing copies. Existing candidate caches can be reselected with
  `calibrate_pid.py --reselect-only`.
- Define the PPO gain box from the validated nominal and robust gains plus a
  `15%` margin. This avoids exposing PPO to the entire differential-evolution
  search box. The current box remains provisional until nominal recalibration.
- The nominal calibration subset contains no injected actuator delay. The robust
  subset includes stationary delay, noise, combined disturbance, and transient
  combined scenarios using `0.06 s` delay and `0.0003 m` noise.
- A validation-only Kp sweep found that nominal error continued improving above
  the old `Kp=50` search ceiling. Error plateaued around `Kp=250-400`; `Kp=600`
  sharply increased steering variation and wheel saturation. In the robust set,
  `Kp=40` increased error by roughly tenfold and `Kp>=50` caused completion loss.
- Therefore the robust optimum is demonstrably delay-limited, while the nominal
  PID must be jointly recalibrated with an expanded Kp range before calibration
  and PPO scheduler bounds are frozen.

## 2026-08-22: Pragmatic calibration freeze

- Do not spend another multi-hour differential-evolution run pursuing a perfect
  delay-free PID. Freeze the nominal PID at `(250.0, 4.5454, 0.3967)`, retaining
  the jointly calibrated `Ki/Kd` and refining only `Kp` with the boundary sweep.
- On validation, `Kp=250` was within `0.31%` of the lowest error at `Kp=400`
  while reducing steering variation by about `40%`. On all six training paths,
  `Kp=250` had lower error and roughly one quarter of the steering variation of
  `Kp=400`. Both completed every scenario.
- Keep the robust PID at `(29.0026, 4.4669, 2.1370)`. Its calibration explicitly
  includes the selected delay and noise conditions.
- Freeze the scheduler action box at `Kp 15-300`, `Ki 3.5-5.0`, and `Kd 0-3.0`.
  The lower Kp limit is supported by earlier Blind_PPO delay sweeps; its exact
  PID results remain prior pilot evidence because that protocol used different
  paths and disturbance schedules.
- Use absolute gain-rate limits `(20.0, 0.5, 1.5)` per second for `(Kp, Ki, Kd)`.
  A box-relative limiter would become roughly seven times faster merely because
  the Kp box was widened, contradicting the prior finding that gain jitter is
  harmful on delayed plants.
- This is a strong empirical baseline, not a claim of globally optimal PID
  gains. No final-test result was inspected or used for this decision.

## 2026-08-22: Native PPO controller ports

- Keep one shared simulator, path sampler, allocator, timing contract,
  disturbance distribution, and evaluator for both learned controllers.
- Restore the proven Blind_PPO scheduler contract: 170 observations, piecewise
  gain mapping around `(26.0, 0.5, 4.4)`, gain box `Kp 10-50`, `Ki 0-1.5`,
  `Kd 2.5-6`, box-relative rate limit `0.5/s`, unfiltered derivative behavior,
  scheduler reward, and original PPO parameters.
- Restore the proven 50 Hz EndToEnd_RL contract: 130 observations, one steering
  action, `0.2 m` cross-track scaling, `0.1 m` tracking-cost scale, steering-rate
  reward, small initial exploration, and original PPO parameters.
- Controller-specific observation encodings and rewards are intentional because
  the actions have different physical meanings. Neither controller receives
  path preview, path identity, disturbance values, or hidden plant context.
- Preserve the historical training defaults and checkpoint techniques:
  scheduler `300k` decisions with validation selection; direct PPO `1M`
  decisions with best training-rollout reward. Training processes are launched
  independently and use one Torch thread by default.
- The earlier `Kp 15-300` box and absolute rate limits remain provenance for the
  fixed-PID calibration work but are superseded for the PPO scheduler.

## 2026-08-22: Shared physical checkpoint selection and 150 ms delay

- A validation audit showed that training-rollout reward selected a materially
  worse direct-PPO checkpoint than physical validation. On the 20-scenario
  diagnostic subset, the nominal direct model improved from `42.58 mm` to
  `11.30 mm` failure-adjusted error when the final rather than reward-selected
  checkpoint was used; the disturbed model improved from `6.41 mm` to
  `5.84 mm`.
- Both learned arms now select checkpoints on the same fixed 24-scenario
  validation subset: maximize completion first, then minimize mean
  failure-adjusted error. Rewards, observations, PPO parameters, and training
  budgets remain architecture-specific.
- The fixed and scheduled PID arms now both use the already selected `0.01 s`
  derivative filter and kick-free initialization. The previous scheduler-only
  unfiltered behavior caused roughly `510` steering-variation units per second
  and `78-79%` wheel saturation in the validation noise condition.
- Set the declared delay severity and disturbed-training delay range to
  `0.15 s`. The prior `0.06 s` delay produced too little separation and did not
  reproduce the delayed regime where the earlier scheduler showed promise.
- These changes alter the control dynamics, training distribution, and model
  selection rule. Every checkpoint under `artifacts/models/native_ports` is
  therefore superseded development evidence and must not be used in the final
  comparison. Protocol-v2 training writes to `artifacts/models/protocol_v2`.
- The fixed robust PID calibrated at `0.06 s` completed only `4/12` transient
  combined checkpoint scenarios at `0.15 s`, so it is superseded. Instead of
  reopening differential evolution, adopt `(18.0, 0.5, 4.4)` from the earlier
  Blind_PPO 150 ms Kp sweep. A deliberately small confirmation at
  `Kp = 18, 22, 26` completed every transient case; `Kp=18` had the lowest
  failure-adjusted error (`17.32 mm`, versus `23.99 mm` and `34.62 mm`). It also
  completed all 12 nominal checkpoint cases at `1.59 mm` failure-adjusted
  error. This is a pragmatic prior-informed baseline, not a new optimum claim.
- The bounds in `calibration.json` now only contain the two fixed baselines.
  PPO scheduler mapping uses the separate tracked `blind_ppo.json` contract.

## 2026-08-22: Scheduler reward scale, disturbance sampler, and gain box floor

A single-seed protocol-v2 audit showed disturbance training helping the direct
arm but hurting the scheduler: transient combined failure-adjusted error moved
from `11.89 mm` (nominal-trained) to `19.86 mm` (disturbance-trained), while
nominal error moved the other way, from `1.96 mm` to `1.39 mm`. Diagnosis and
the three resulting changes are recorded here. No held-out result was opened.

### Diagnosed cause: a saturated scheduler tracking term

The tracking reward is `x^2/(1+x^2)` with `x = distance / tracking_scale_m`.
The native Blind_PPO scheduler scale is `0.005 m`. Measured gradient of that
term is `130 /m` at `3 mm` error but `1.2 /m` at `34 mm`. Under the `0.15 s`
delay the plant tracks at `22-34 mm`, so the term sat pinned at `0.97-0.98`
and the scheduler's disturbed episodes contributed almost no tracking
gradient; `PROGRESS_WEIGHT = 5.0` and the terminal bonus dominated instead.

The remaining usable gradient came from the near-nominal episodes, and a fixed
gain sweep on the 12 transient combined validation scenarios shows those pull
`Kp` the wrong way. Nominal error falls monotonically in `Kp` (`2.79 mm` at
`Kp=6` to `1.25 mm` at `Kp=50`) while transient combined error rises steeply
(`9.38 mm` at `Kp=8`, `17.32 mm` at `Kp=18`, `34.62 mm` at `Kp=26`, with
completion loss from `Kp=32`). The disturbance-trained scheduler settled at
`Kp ~21-27` and the nominal-trained one at `Kp ~11-13`.

The direct arm uses a `0.1 m` scale, so its `16-44 mm` errors stayed inside the
responsive part of the same curve, and its disturbance training worked. Same
plant, same sampler, same severities, opposite outcome. That isolates the
tracking scale rather than the disturbance distribution as the primary cause.

### Changes

- Set the scheduler `tracking_scale_m` from `0.005` to `0.02`. At `0.02` both
  the nominal operating point (`3 mm`) and the disturbed one (`25 mm`) keep a
  live gradient. This supersedes the native Blind_PPO reward scale; every other
  scheduler reward constant is unchanged.
- Replace the disturbance-training sampler. It now mirrors the four evaluation
  conditions with weights `(nominal 0.10, delay 0.30, noise 0.20, combined
  0.40)`, takes the stationary or transient form with equal probability, draws
  the declared evaluation severity outright half the time, and steps delay and
  noise together in the combined transient exactly as the evaluation manifest
  builds them. Event times are sampled as a fraction of the episode's own ideal
  duration and resolved in `reset`, spanning `1.0 s` to `0.8 x` ideal duration.
  The previous sampler drew delay uniformly and then zeroed it whenever it
  added an event, giving a mean sampled delay of `0.0366 s` against a `0.15 s`
  evaluation severity, never co-occurring delay and noise steps, and a
  hardcoded `6-12 s` event window derived from an assumed `20 s` episode when
  actual training episodes run `9-53 s`. The replacement gives a mean applied
  delay of `0.0787 s` with `34.8%` of episodes at the declared severity.
- Lower the scheduler gain box floor from `Kp 10` to `Kp 5`. The nominal-trained
  scheduler pinned itself at `10.00`, the box floor, and the fixed sweep puts
  the transient combined optimum near `Kp 8`, below the old floor. The box came
  from the Blind_PPO port under the superseded `0.06 s` delay. The box-relative
  rate limit follows the widened box, from `20.0` to `22.5` per second for `Kp`.

### Open item recorded before the changes

The robust fixed PID `(18.0, 0.5, 4.4)` was selected from a three-point
confirmation at `Kp = 18, 22, 26`, and `18` was the lowest value tested. The
sweep above reaches `9.38 mm` at `Kp=8` against its `17.32 mm`, so the robust
baseline is under-tuned and every learned advantage over it is inflated.
Re-tune it downward before the freeze. Related: the nominal-trained scheduler's
`11.89 mm` is not distinguishable from a static `Kp=13` PID at `11.80 mm`, so
no adaptation claim is supported by the current evidence.

All four protocol-v2 seed-11 checkpoints predate these changes and are
superseded development evidence.
