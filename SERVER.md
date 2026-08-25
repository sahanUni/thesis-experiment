# Running the frozen matrix on the compute server

The laptop stays the interactive machine: dashboard, MuJoCo replay, analysis.
The server is for the frozen five-seed matrix only. Nothing here renders.

`PLAN.md` requires final results to come from **one declared machine**. If the
matrix runs here, the fixed-PID arms must be evaluated here too. Server PPO
results may not be mixed with laptop PID numbers in one result table.

## 1. Get the code across

```bash
ssh <your-account>@login01.int.scc.plus.ac.at
git clone https://github.com/sahanUni/thesis-experiment.git
cd thesis-experiment
```

### Building Python 3.12 with uv (no root needed)

The cluster's system Python is older than 3.12 and cannot be upgraded without
root. `uv` downloads its own standalone interpreter into your home directory,
which needs no privileges:

```bash
export RAYON_NUM_THREADS=2          # login01 caps threads per user
curl -LsSf https://astral.sh/uv/install.sh | sh    # only if uv is missing
source $HOME/.local/bin/env

uv python install 3.12
cd ~/thesis-experiment
uv venv --python 3.12 .venv
source .venv/bin/activate
python -V                           # must say 3.12.x
uv pip install -r requirements.txt
python tools/show_versions.py
```

`uv pip` does not need `ensurepip`, so this also avoids the missing-pip problem
below.

Two constraints. Build the venv under `$HOME` or another shared filesystem: a
venv on node-local `/tmp` will not exist after the next `srun`. And export
`RAYON_NUM_THREADS` before any `uv` command on the login node, or uv fails with
`failed to initialize global rayon pool` against the per-user thread cap.

### Reusing an existing Python 3.12 venv

If Python 3.12 was built by hand, its venv may have been created without pip.
Activate the venv and bootstrap pip in place:

```bash
source $HOME/venv312/bin/activate     # your existing venv
python -V                             # must say 3.12.x
echo "VIRTUAL_ENV=$VIRTUAL_ENV"       # must not be empty
python -m ensurepip --upgrade         # when the pip command is missing
python -m pip --version
```

`srun --pty bash` opens a **fresh shell on the compute node**, so a venv
activated on the login node does not carry over. Activate it again after every
`srun`. If you skip this, pip falls back to the system interpreter and reports
`site-packages is not writeable`, then fails to resolve `numpy==2.2.6` while
offering only versions old enough for that Python. Calling the interpreter by
path avoids the ambiguity entirely:

```bash
$HOME/venv312/bin/python -m pip install -r requirements.txt
```

Use `python -m pip`, never bare `pip`: bare `pip` resolves through `PATH` and
will pick the system one again.

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

Those are the laptop's numbers. **Completion must match exactly** -- it is
discrete and it is the primary outcome. The errors are checked within a 1%
relative band, because MuJoCo is not bit-identical across CPU architectures and
a last-bit difference amplifies over a 20-50 s closed-loop episode. Measured
drift from the laptop to `cpunode13` was 0.01% to 0.34%, well inside the band.

`pid_global_nominal` under the transient combined condition is checked on
completion only. It fails 0/12, so its `J_FA` is dominated by
`corridor * (T_max - t_end)` -- a discontinuous term that moves by percent when
the corridor exit shifts by a millisecond. The finding there is that it fails,
not the exact number.

Tighten or loosen the band with `--tolerance` if you need to. A genuine plant
difference shows up as changed completion or as whole-percent error changes,
not as a fourth decimal place.

PPO arms are excluded from the check entirely. Torch on a different CPU can
reorder floating-point reductions and shift an action in the last bits.

This is a sanity check, not a reproducibility guarantee. `PLAN.md` handles
cross-machine variation by requiring every final result to come from one
declared machine, which is why the fixed-PID arms must be re-evaluated here
rather than carried over from the laptop.

## 3. This is a SLURM cluster

`sinfo -s` shows one partition, `base`, 48 nodes, no time limit. `login01` is
for editing and submitting only -- never for training. It is tightly capped on
threads per user.

### Interactive, inside tmux (the usual route)

tmux keeps the allocation alive across disconnects. **Start tmux on the login
node, then take the allocation inside it** -- that order matters. tmux started
from inside an allocation puts its server on the compute node, where it dies
with the allocation and cannot be reached, because compute nodes do not accept
direct ssh.

```bash
tmux new -s thesis
srun --partition=base --cpus-per-task=20 --mem=40G --constraint=compute_nodes_cpu \n  --time=12:00:00 --pty bash
# now on a compute node
cd ~/thesis-experiment && source .venv/bin/activate
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

### Reattaching, and why the node changes

The tmux session lives on `login01`, not on the compute node. The `srun` shell
inside it runs on whichever node SLURM gave you. So you reconnect through the
login node:

```bash
ssh <your-account>@login01.int.scc.plus.ac.at
tmux attach -t thesis
```

You never ssh to `cpunode13` directly, and you do not need to. But the tmux
session is only as durable as `login01`: if it reboots or the session is
killed, the allocation and every running process go with it.

The node itself will differ next time. That does not affect your files -- home
is shared ceph, visible from every node -- but it does affect numbers. Two
different CPUs produce slightly different trajectories, which is exactly the
drift `check_parity.py` measures. So keep one result set to one node:

- **Training and evaluation belong in the same job.** `slurm_batch.sh` runs
  parity, all twenty runs, and the held-out evaluation in one allocation, so
  they cannot land on different nodes.
- If you must split them, pin the node with `--nodelist=cpunode13`, or check
  whether the partition is homogeneous first:
  `sinfo -o "%n %c %m %f" | sort -u -k2`.
- `evaluate.py` records the node, CPU model, and SLURM job id under `machine`
  in `metadata.json`, so a result set that accidentally spans nodes is
  detectable afterwards rather than silently wrong.

### Batch, fire and forget

`slurm_batch.sh` runs the parity check, the twenty training runs, and the
held-out evaluation as one job. It needs no tmux at all and survives a lost
login node, which makes it the better choice for a 5.5 hour batch:

```bash
sbatch slurm_batch.sh                          # uses ./.venv by default
# or, for a venv outside the repo:
# sbatch --export=ALL,VENV=$HOME/venv312 slurm_batch.sh
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

The nodes carry 128 cores and 2.3 TB each, so 20 is a modest request:

```
$ sinfo -o "%n %c %m %f" | sort -u -k2
cpunode08.dyn.scc.plus.ac.at 128 2321340 compute_nodes_cpu
gpunode09.dyn.scc.plus.ac.at 128 2321340 compute_nodes_gpu
```

Beyond 20 the extra cores idle, because there are only 20 runs. Use
`--constraint=compute_nodes_cpu` to keep off the gpunodes: they report the same
core count but may carry a different CPU model, and two CPUs produce slightly
different trajectories.

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
rsync -av SERVER:thesis-experiment/artifacts/results/final ./artifacts/results/
rsync -av --include="*/" --include="metadata.json" --include="validation_history.jsonl" --include="monitor.csv" --include="*.zip" --exclude="*" SERVER:thesis-experiment/artifacts/models/final ./artifacts/models/
```

Model zips are about 1 MB each and are worth having for the dashboard. The
tensorboard directories are the bulk and are rarely worth moving.

Then, on the laptop:

```powershell
..\venv\Scripts\python.exe analyze.py --episodes artifacts\results\final\episodes.csv
..\venv\Scripts\python.exe dashboard.py
```

## Troubleshooting

- **Activating an old venv fails with a permission error** -- read
  `<venv>/pyvenv.cfg`. It names the interpreter the venv was built from, and a
  venv breaks when that interpreter is removed or becomes unreadable rather
  than when the venv itself changes. Rebuild with uv above; it does not depend
  on a system interpreter staying put.
- **`pip: command not found`** -- the venv was created without pip. See
  section 1: `python -m ensurepip --upgrade`, then `python -m pip`.
- **`Defaulting to user installation because normal site-packages is not
  writeable`, then `No matching distribution found for numpy==2.2.6`** -- the
  venv is not active and pip is running on the system Python. The version list
  pip offers tells you which: numpy stops at 2.0.2 on Python 3.9 and at 2.2.6
  on 3.10. Re-activate the venv, or call
  `$HOME/venv312/bin/python -m pip` by path. `python tools/show_versions.py`
  reports the interpreter and refuses to pass outside a virtualenv.
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
