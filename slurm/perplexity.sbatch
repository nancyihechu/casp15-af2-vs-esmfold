#!/bin/bash
#SBATCH --job-name=perplexity
#SBATCH --partition=gpu
#SBATCH --account=dkhf3-lab
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# Run scripts/perplexity.py on all targets, in the esmfold env, offline.
# Other target set: sbatch --export=ALL,CONFIG=config/positive_control.yaml slurm/perplexity.sbatch
set -eo pipefail
cd /cluster/pixstor/dkhf3-lab/users/nancy/PROJECT3_TFM
module load miniconda3/v26.7.1-1_py314
eval "$(conda shell.bash hook)"
conda activate /cluster/pixstor/dkhf3-lab/users/nancy/envs/esmfold
export HF_HOME=/cluster/pixstor/dkhf3-lab/users/nancy/hf_cache
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name --format=csv,noheader
python scripts/perplexity.py ${CONFIG:+--config $CONFIG}
