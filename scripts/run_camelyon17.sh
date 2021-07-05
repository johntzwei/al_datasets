#!/bin/bash
#SBATCH --job-name=al_wilds
#SBATCH --gres=gpu:2080:1
#SBATCH --ntasks=1
#SBATCH --array=1-3

source activate al_datasets

TOKENIZERS_PARALLELISM=false \
python al_wilds.py \
    -d camelyon17 \
    --algorithm ERM \
    --root_dir ../data \
    --progress_bar \
    --log_dir ../experiments/camelyon17/ \
    --evaluate_all_splits False \
    --al_strategy us \
    --total_data 50000 \
    --seed 2 \
    --exp_id 0 \
    --rounds 5 \
    --download

#    --seed $SLURM_ARRAY_TASK_ID \
#    --exp_id rs_$SLURM_ARRAY_TASK_ID \

