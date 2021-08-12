#!/bin/bash
#SBATCH --job-name=al_wilds
#SBATCH --gres=gpu:2080:1
#SBATCH --ntasks=1
#SBATCH --array=1-3

source activate al_datasets

TOKENIZERS_PARALLELISM=false \
python wilds_al_expt.py \
    -d civilcomments \
    --algorithm ERM \
    --model roberta-base \
    --root_dir ../data \
    --progress_bar \
    --log_dir ../experiments/civilcomments/ \
    --evaluate_all_splits False \
    --al_strategy us \
    --total_data 1000 \
    --valid_size 1000 \
    --val_metric acc_avg \
    --patience 2 \
    --frac 0.1 \
    --seed 10 \
    --exp_id us_train \
    --rounds 1 \
    --download

#    --seed $SLURM_ARRAY_TASK_ID \
#    --exp_id rs_$SLURM_ARRAY_TASK_ID \

