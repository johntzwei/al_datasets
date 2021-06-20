#!/bin/bash
#SBATCH --job-name=al_wilds
#SBATCH --gres=gpu:2080:1
#SBATCH --ntasks=1
#SBATCH --array=1-3

source activate al_datasets

TOKENIZERS_PARALLELISM=false \
python run_expt.py \
    -d civilcomments \
    --algorithm ERM \
    --root_dir data \
    --progress_bar \
    --log_dir civilcomments_logs \
    --evaluate_all_splits False \
    --al_strategy rs \
    --total_data 50000 \
    --rounds 5 \
    --seed $SLURM_ARRAY_TASK_ID \
    --exp_id rs_$SLURM_ARRAY_TASK_ID \
    --download
