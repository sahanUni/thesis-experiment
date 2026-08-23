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
```

### Using an existing Python 3.12 venv

If Python 3.12 was built by hand, its venv may have been created without pip.
Activate the venv and bootstrap pip in place:

```bash
source $HOME/venv312/bin/activate     # your existing venv
python -V                             # confirm 3.12
python -m ensurepip --upgrade         # when the pip command is missing
python -m pip --version
```

If `ensurepip` is absent too, the interpreter was built without it. Bootstrap
once from the standalone installer:

```bash
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py
```

Check what the venv already has before installing anything. This needs no pip:

```bash
python tools/show_versions.py
```

Compare against `requirements.txt`. The `numpy` and `mujoco` versions are the
ones that matter -- they are the physics, and a different patch version fails
the parity check in section 2. Install or correct whatever differs:

```bash
python -m pip install -r requirements.txt
```

Or build a fresh venv instead, if you would rather not disturb the existing one:

```bash
python -m venv venv && source venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins `numpy`, `scipy`, `mujoco`, `gymnasium`,
`stable-baselines3` and `torch` exactly. Keep it that way.

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

### Interactive, inside tmux (the usual route)

tmux keeps the allocation alive across disconnects. Start tmux on the login
node, then take the allocation inside it:

```bash
tmux new -s thesis
srun --partition=base --cpus-per-task=6 --mem=16G --time=12:00:00 --pty bash
# now on a compute node
cd ~/Thesis_Experiment && source $HOME/venv312/bin/activate
python check_parity.py && python run_seeds.py --phase final
# Ctrl-b then d to detach; tmux attach -t thesis to return
```

The catch: if the tmux session dies, the allocation and the runs die with it.
`run_seeds.py` writes each run's output to
`artifacts/models/final/<arm>/seed_<N>/train.log`, so the record survives a
lost terminal even though the processes do not.

After training, still inside the allocation:

```bash
python evaluate.py --manifest manifests/held_out.json --models-root artifacts/models/final --run-id final
```

### Batch, fire and forget

`slurm_batch.sh` runs the parity check, the twenty training runs, and the
held-out evaluation as one job. It survives a lost tmux session:

```bash
sbatch --export=ALL,VENV=$HOME/venv312 slurm_batch.sh
squeue -u $USER
tail -f slurm-thesis-final-<jobid>.out
```

### Core budget and expected wall time

One process per (arm, seed), single-threaded. Env stepping and the serial PPO
gradient update split training wall time roughly evenly, so parallel
environments inside one run saturate quickly while independent runs scale close
to linearly.

The matrix is 20 runs: 10 scheduler arms at 300k decisions (about 40 min each)
and 10 direct arms at 2M (about 2.5 h each), so roughly 32 core-hours.

| `--cpus-per-task` | expected wall time |
|---:|---|
| 6 | about 5.5 h |
| 12 | about 3 h |
| 20 | about 2.5 h, bounded by one direct run |

`run_seeds.py` starts the long direct runs first so the short scheduler runs
fill the gaps as cores free up. Starting them in declaration order would leave
cores idle at the end and costs about an hour on six cores. Six cores fits
inside the 12 h limit with margin, so it is a safe default, but raising it
shortens the batch nearly linearly up to 20.

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
rsync -av --include="*/" --include="metadata.json" --include="validation_history.jsonl" --include="monitor.csv" --include="*.zip" --exclude="*" SERVER:Thesis_Experiment/artifacts/models/final ./artifacts/models/
```

Model zips are about 1 MB each and are worth having for the dashboard. The
tensorboard directories are the bulk and are rarely worth moving.

Then, on the laptop:

```powershell
..\venv\Scripts\python.exe analyze.py --episodes artifacts\results\final\episodes.csv
..\venv\Scripts\python.exe dashboard.py --results artifacts\results\final
```

## Troubleshooting

- **`pip: command not found`** -- the venv was created without pip. See
  section 1: `python -m ensurepip --upgrade`, then `python -m pip`.
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
- **`failed to initialize global rayon pool`** -- the login node caps threads
  per user. Do not run work there; take an allocation with `srun` first.
