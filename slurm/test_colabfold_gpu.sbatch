#!/bin/bash
#SBATCH --job-name=cf_gputest
#SBATCH --partition=gpu
#SBATCH --account=dkhf3-lab
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x-%j.out
# Check that JAX in envs/colabfold sees the GPU.
E=/cluster/pixstor/dkhf3-lab/users/nancy/envs/colabfold
nvidia-smi --query-gpu=name --format=csv,noheader
$E/bin/python -c "import jax; print('JAX devices:', jax.devices())"
