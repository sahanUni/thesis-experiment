# Running the frozen matrix on the compute server

The laptop stays the interactive machine: dashboard, MuJoCo replay, analysis.
The server is for the frozen five-seed matrix only. Nothing here renders.

`PLAN.md` requires final results to come from **one declared machine**. If the
matrix runs here, the fixed-PID arms must be evaluated here too. Server PPO
results may not be mixed with laptop PID numbers in one result table.

## 1. Get the code across

```bash
ssh <your-account>@login01.int.scc.plus.ac.at
git clone <repo-url> Thesis_Experiment
cd Thesis_Experiment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `numpy`, `scipy`, `mujoco`, `gymnasium`,
`stable-baselines3` and `torch` exactly. Keep it that way. MuJoCo is the
physics, and a different patch version moves trajectories enough to fail the
parity check below.

The clone carries `artifacts/calibration/*.json` (the frozen gains and both
action boxes -- experiment INPUTS, not outputs), the manifests, and
`artifacts/parity/reference.json`. Everything else under `artifacts/` is
ignored by git.

## 2. Check the machine agrees with the laptop, before spending compute on it

```bash
python -m pytest -q
python check_parity.py
```

The two fixed-PID arms contain no neural network, so their numbers are a pure
function of the physics, the geometry, and the calibration artifact. They must
match the reference **exactly**:

| arm | gains | condition | completion | J_FA |
|---|---|---|---:|---:|
| `pid_global_nominal` | `(250, 4.5454, 0.3967)` | nominal | 1.00 | `1.1586 mm` |
| `pid_global_nominal` | | transient combined | 0.00 | `151.9646 mm` |
| `pid_global_robust` | `(8.0, 0.5, 4.4)` | nominal | 1.00 | `2.3542 mm` |
| `pid_global_robust` | | transient combined | 1.00 | `9.3779 mm` |

`check_parity.py` exits non-zero on any difference. A mismatch is a plant
difference, not noise: check the `mujoco` and `numpy` versions first. Do not
train until it passes.

PPO arms are deliberately excluded from the check. Torch on a different CPU can
reorder floating-point reductions and shift an action in the last bits. That is
expected and is not evidence of a broken plant.

## 3. This is a SLURM cluster

`sinfo -s` shows one partition, `base`, 48 nodes, no time limit. `login01` is
for editing and submitting only -- never for training. It is tightly capped on
threads per user.

**Batch, fire and forget.** `slurm_batch.sh` runs the parity check, the twenty
training runs, and then the held-out evaluation:

```bash
sbatch slurm_batch.sh
squeue -u $USER
tail -f slurm-thesis-final-<jobid>.out
```

Override the output root without editing the script:

```bash
sbatch --export=ALL,OUTPUT=artifacts/models/final_rerun slurm_batch.sh
```

**Interactive, inside tmux**, when you want to watch the first minutes:

```bash
tmux new -s thesis
srun --partition=base --cpus-per-task=20 --mem=40G --time=12:00:00 --pty bash
cd ~/Thesis_Experiment && source venv/bin/activate
python check_parity.py && python run_seeds.py --phase final
# Ctrl-b then d to detach; tmux attach -t thesis to return
```

The catch: if the tmux session dies, the allocation and the runs die with it.

### Core budget

`--cpus-per-task=20` for twenty runs: one core each. One process per
(arm, seed), single-threaded. Env stepping and the serial PPO gradient update
split training wall time roughly evenly, so parallel environments inside one
run saturate quickly while independent runs scale close to linearly.

The scheduler arms are 300k decisions (about 40 min each) and the direct arms
2M (about 2.5 h each), so with all twenty concurrent the batch finishes in
roughly the wall time of one direct run. Asking for a whole node would not
finish sooner.

`os.cpu_count()` inside a job reports the whole compute node rather than the
allocation, so `run_seeds.py` reads `SLURM_CPUS_PER_TASK` and falls back to
`os.sched_getaffinity`.

## 4. Threads

`--threads-per-job` defaults to 1, and the launcher exports `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and
`VECLIB_MAXIMUM_THREADS` to match. Without this every process grabs every core,
and twenty runs each spawning twenty threads on twenty cores is far slower than
twenty single-threaded ones.

Thread count changes floating-point reduction order, so keep it fixed across
runs that are meant to be compared.

## 5. Collect the results

`slurm_batch.sh` already runs the held-out evaluation and writes
`artifacts/results/final/`. Copy back, from the laptop:

```bash
rsync -av SERVER:Thesis_Experiment/artifacts/results/final ./artifacts/results/
rsync -av --include="*/" --include="metadata.json" --include="validation_history.jsonl" --include="monitor.csv" --exclude="*" SERVER:Thesis_Experiment/artifacts/models/final ./artifacts/models/
```

Model zips are about 1 MB each if you want them for the dashboard. The
tensorboard directories are the bulk and are rarely worth moving.

Then, on the laptop:

```powershell
..\venv\Scripts\python.exe analyze.py --episodes artifacts\results\final\episodes.csv
..\venv\Scripts\python.exe dashboard.py --results artifacts\results\final
```

## Troubleshooting

- **`invalid value for environment variable MUJOCO_GL`** -- something set it to
  a backend this machine lacks. The launcher deliberately leaves it unset,
  because training never renders. If a headless box still complains,
  `export MUJOCO_GL=osmesa`.
- **Runs die instantly** -- read
  `artifacts/models/final/<arm>/seed_<N>/train.log`; the launcher captures
  stdout and stderr there and reports a non-zero exit per run.
- **Parity fails on `mean_distance_m` but not `completion_rate`** -- a small
  numeric drift, almost always a `mujoco` patch version. Reinstall from the
  pinned `requirements.txt`.
- **`evaluate.py` refuses to write** -- it will not overwrite a non-empty result
  directory. Pass a new `--run-id` rather than deleting a sealed one.
