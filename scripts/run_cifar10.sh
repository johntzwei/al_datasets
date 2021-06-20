#!/bin/bash
#SBATCH --job-name=al_cifar
#SBATCH --gres=gpu:1080:1
#SBATCH --ntasks=1
#SBATCH --array=1-3

source activate al_datasets

python cifar10.py \
    -e 3 \
    -p /home/johnny/al_datasets/experiments/cifar10/ \
    -s uncertainty \
    --valid_size 1000 \
    --total_data 20000 \
    --batch_size 64 \
    --epochs 50 \
    --patience 5

#    -e $SLURM_ARRAY_TASK_ID \
