#!/bin/bash
#SBATCH --job-name=al_wilds
#SBATCH --gres=gpu:2080:1
#SBATCH --ntasks=1
#SBATCH --array=1-3

TOKENIZERS_PARALLELISM=false \
python al_wilds.py \
    -d amazon \
    --algorithm ERM \
    --root_dir data \
    --progress_bar \
    --log_dir amazon_logs \
    --evaluate_all_splits False \
    --al_strategy us \
    --total_data 50000 \
    --rounds 5 \
    --seed 0 \
    --exp_id 0 \
    --download

#    --seed $SLURM_ARRAY_TASK_ID \
#    --exp_id us_$SLURM_ARRAY_TASK_ID \

