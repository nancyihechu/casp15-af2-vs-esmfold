#!/bin/bash
#SBATCH --job-name=af2_sweep
#SBATCH --partition=gpu
#SBATCH --account=dkhf3-lab
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm/logs/%x-%A_%a.out
# AlphaFold2 (ColabFold) on all MSA depths, one array task per target (index in config/targets.yaml).
# Usage: sbatch slurm/af2_array.sbatch            (all targets)
#        sbatch --array=0 slurm/af2_array.sbatch  (first target only)
set -euo pipefail
cd /cluster/pixstor/dkhf3-lab/users/nancy/PROJECT3_TFM
export HF_HUB_OFFLINE=1
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=4.0
nvidia-smi --query-gpu=name --format=csv,noheader
/cluster/pixstor/dkhf3-lab/users/nancy/envs/p3_tools/bin/python scripts/run_af2.py --index "$SLURM_ARRAY_TASK_ID" ${DEPTHS:+--depths $DEPTHS}
