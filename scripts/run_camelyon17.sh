#!/bin/bash
#SBATCH --job-name=al_wilds
#SBATCH --gres=gpu:2080:1
#SBATCH --ntasks=1
#SBATCH --array=0-9

source activate al_datasets

TOKENIZERS_PARALLELISM=false \
python wilds_al_expt.py \
    -d camelyon17 \
    --algorithm ERM \
    --root_dir ../data \
    --progress_bar \
    --log_dir ../experiments/camelyon17/ \
    --evaluate_all_splits False \
    --al_strategy us \
    --total_data 30000 \
    --seed $SLURM_ARRAY_TASK_ID \
    --exp_id us_$SLURM_ARRAY_TASK_ID \
    --rounds 10 \
    --valid_size 3000 \
    --patience 10 \
    --download

