#!/bin/bash
#SBATCH --job-name=al_wilds
#SBATCH --gres=gpu:2080:1
#SBATCH --ntasks=1
#SBATCH --array=1-3

source activate al_datasets

TOKENIZERS_PARALLELISM=false \
python al_wilds.py \
    -d py150 \
    --algorithm ERM \
    --root_dir ../data \
    --progress_bar \
    --log_dir ../experiments/py150/ \
    --evaluate_all_splits False \
    --al_strategy us \
    --total_data 10000 \
    --seed 0 \
    --exp_id 0 \
    --rounds 1 \
    --download

#    --seed $SLURM_ARRAY_TASK_ID \
#    --exp_id rs_$SLURM_ARRAY_TASK_ID \

