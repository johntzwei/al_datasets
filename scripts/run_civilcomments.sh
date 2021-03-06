#!/bin/bash
#SBATCH --job-name=al_wilds
#SBATCH --gres=gpu:2080:1
#SBATCH --ntasks=1
#SBATCH --array=1-9

source activate al_datasets

TOKENIZERS_PARALLELISM=false \
python wilds_al_expt.py \
    -d civilcomments \
    --algorithm ERM \
    --model distilbert-base-uncased \
    --root_dir ../data \
    --progress_bar \
    --log_dir ../experiments/civilcomments/ \
    --evaluate_all_splits False \
    --al_strategy us \
    --total_data 20000 \
    --valid_size 2000 \
    --val_metric acc_avg \
    --patience 10 \
    --seed $SLURM_ARRAY_TASK_ID \
    --exp_id us_$SLURM_ARRAY_TASK_ID \
    --rounds 10 \
    --download

