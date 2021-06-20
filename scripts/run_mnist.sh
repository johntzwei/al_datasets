#!/bin/bash
#SBATCH --job-name=al_mnist
#SBATCH --gres=gpu:1080:1
#SBATCH --ntasks=1
#SBATCH --array=1-3

source activate al_datasets

python mnist.py \
    -e 3 \
    -p /home/johnny/al_datasets/experiments/mnist/ \
    -s uncertainty \
    --valid_size 100 \
    --total_data 1000 \
    --batch_size 8 \
    --epochs 50 \
    --patience 10

#    -e $SLURM_ARRAY_TASK_ID \
