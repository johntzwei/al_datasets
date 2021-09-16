#!/bin/bash
#SBATCH --job-name=mnli_grid
#SBATCH --gres=gpu:2080:1
#SBATCH --ntasks=1
#SBATCH --array=1-15

source activate al_datasets

TOKENIZERS_PARALLELISM=false \
python wilds_grid_expt.py \
    -d mnli \
    --algorithm ERM \
    --model roberta-base \
    --root_dir ../data \
    --progress_bar \
    --log_dir ../experiments/civilcomments_grid/ \
    --evaluate_all_splits False \
    --val_metric acc_avg \
    --exp_id ss_$SLURM_ARRAY_TASK_ID \
    --patience 3 \
    --minority contains_negation \
    --majority contains_negation \
    --groupby_fields contains_negation y \
    --grid_index $SLURM_ARRAY_TASK_ID \
    --ex_step_size 2000 \
    --num_grid_rc 4 \
    --download
