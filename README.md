# Thesis Experiment

Final fixed-PID versus PPO gain-scheduling versus direct-PPO path-following
experiment. All controller arms use the same MuJoCo plant, allocator, speed PID,
scenario schema, causal feedback information, 500 Hz physics, and 50 Hz
learned-policy rate. Blind PPO keeps its native 170-value history and direct PPO
keeps its native 130-value history.

`PLAN.md` is the scientific protocol. `TODO.md` is the gated execution list.
`DECISIONS.md` records changes made before and after the final freeze.

## Setup

From `D:\Msc\Experiments\Thesis_Experiment`:

```powershell
..\venv\Scripts\python.exe -m pip install -r requirements.txt
..\venv\Scripts\python.exe -m pytest -q
..\venv\Scripts\python.exe make_manifests.py
```

The disturbance severities, fixed PID calibration, and native PPO controller
profiles are recorded in `DECISIONS.md`.

```powershell
..\venv\Scripts\python.exe probe_sensitivity.py --probe all --development-default
```

## Execution Flow

1. The calibration below is historical and should only be rerun if the PID
freeze is deliberately reopened. It overwrites the tracked calibration artifact:

```powershell
..\venv\Scripts\python.exe calibrate_pid.py --per-path
```

To reapply the selection rule and validation to existing candidate caches
without rerunning differential evolution:

```powershell
..\venv\Scripts\python.exe calibrate_pid.py --reselect-only --per-path
```

The validation-only Kp boundary diagnostic used for the current freeze is:

```powershell
..\venv\Scripts\python.exe probe_pid_boundary.py
```

2. Print the four one-seed development commands, then execute them:

```powershell
..\venv\Scripts\python.exe run_training_matrix.py --phase development
```

The command prints four independent processes. Run them in separate terminals
to train concurrently. Add `--execute` only for sequential execution.

3. Debug only with `manifests/train.json` and `manifests/validation.json`. Freeze
the code, budget, gains, severities, manifests, and seeds before the final run.

4. Run the frozen five-seed matrix:

```powershell
..\venv\Scripts\python.exe run_training_matrix.py --phase final --execute
```

5. Evaluate the complete model matrix on the held-out manifest:

```powershell
..\venv\Scripts\python.exe evaluate.py `
  --manifest manifests\held_out.json `
  --models-root artifacts\models\protocol_v2 `
  --run-id final
```

6. Generate paired effect intervals:

```powershell
..\venv\Scripts\python.exe analyze.py `
  --episodes artifacts\results\final\episodes.csv
```

`analyze.py` is where official numbers come from. The dashboard is the
exploratory counterpart: it builds a `scenarios.Scenario` from the drawer
controls and runs it on demand through the same `rollout.run_episode` the
evaluator uses, so a scenario replayed there reproduces the batch trace sample
for sample -- asserted by `tests/test_dashboard.py`. It writes nothing.

```powershell
..\venv\Scripts\python.exe dashboard.py
```

Fixed PID, scheduled PPO and direct RL are picked one row each in the drawer
and overlaid on shared axes: tracking error, the dead time actually in force,
the steering yaw rate delivered to the plant, the wheel differential against
its saturation bounds, and Kp. PPO artifacts are discovered under
`--models-root` (default `artifacts/models/final`), and `--calibration`
defaults to `artifacts/calibration/calibration.json`, falling back to the
development gain box with a visible warning when that file does not exist yet.

## Artifacts

- `manifests/`: serialized paired scenarios and RNG seeds.
- `artifacts/calibration/`: PID gains, scheduler bounds, and all candidate rows.
- `artifacts/models/protocol_v2/<arm>/seed_<seed>/`: checkpoints, monitor data, validation
  history, configuration, package versions, and timing.
- `artifacts/models/native_ports/`: superseded development checkpoints retained
  only as diagnostic provenance; they are incompatible with protocol v2.
- `artifacts/results/<run-id>/`: immutable manifest/calibration copies, episode
  table, aggregate table, hashes, traces, and paired-effect tables.

Development defaults are never thesis results. Official results require the
freeze gate, all five declared seeds, deterministic policy actions, and the
complete paired result matrix.

`held_out.json` is the primary generated-path test. `structural_ood.json` and
`stress.json` are separate appendix evaluations and must not be pooled with it.
Use `--save-traces` on a small, preselected representative manifest when 500 Hz
dashboard traces are needed; the full evaluator otherwise stores episode rows
without duplicating high-rate trajectories for every model and scenario.
