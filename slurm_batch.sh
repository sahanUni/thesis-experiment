#!/bin/bash
# Frozen five-seed PPO matrix as a SLURM batch job.
#
#   sbatch slurm_batch.sh
#   sbatch --export=ALL,VENV=$HOME/venv312,OUTPUT=artifacts/models/final slurm_batch.sh
#
# One process per (arm, seed), one thread each: 4 arms x 5 seeds = 20 runs.
# Env stepping and the serial PPO gradient update split training wall time
# roughly evenly, so parallel environments inside one run saturate quickly
# while independent runs scale close to linearly.
#
# With 6 cores the 20 runs are queued 6 at a time. run_seeds.py starts the 2M
# direct runs first so the short scheduler runs fill the gaps as cores free up;
# expect roughly 5.5 hours. Raising --cpus-per-task shortens that nearly
# linearly up to 20, where the batch takes about the wall time of one direct
# run.

#SBATCH --job-name=thesis-final
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

VENV="${VENV:-venv}"
OUTPUT="${OUTPUT:-artifacts/models/final}"
SAMPLER="${SAMPLER:-dynamic}"
PHASE="${PHASE:-final}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# run_seeds.py exports these to its children too; setting them here covers
# anything that reads them before the launcher starts. MUJOCO_GL is left unset
# deliberately: training never renders, and mujoco only needs a backend when a
# rendering context is actually created.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none} on ${SLURM_JOB_PARTITION:-none}"
echo "cores     : ${SLURM_CPUS_PER_TASK:-unknown} allocated"
echo "python    : $(python -V 2>&1)"
echo "venv      : ${VENV}"
echo "output    : ${OUTPUT}"
echo

echo "=== parity check: the plant must match the declared machine ==="
python check_parity.py
echo

echo "=== training matrix ==="
python run_seeds.py \
  --phase "${PHASE}" \
  --output "${OUTPUT}" \
  --sampler-profile "${SAMPLER}" \
  --threads-per-job 1

echo
echo "=== held-out evaluation ==="
python evaluate.py \
  --manifest manifests/held_out.json \
  --models-root "${OUTPUT}" \
  --run-id final

echo
echo "done. Copy artifacts/results/final and the metadata.json files back to the laptop."
echo "Model zips are ~1 MB each; tensorboard directories are the bulk and rarely worth moving."
