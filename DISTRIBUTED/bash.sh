#!/bin/bash
# SLURM submit script: 2 nodes x 2 A100 GPUs (adapt the resources to your cluster)
#SBATCH -N 2
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=2      # one task per GPU
#SBATCH -c 32
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --output=slurm-%j.out

source $STORE/mypython/bin/activate   # environment with requirements.txt installed

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

# Download the dataset, model and metric on a login node first if compute nodes have no internet.
srun python train_ddp.py --nodes 2 --devices 2 --output results_2x2.json
