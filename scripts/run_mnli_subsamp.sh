#!/bin/bash
#SBATCH --job-name=mnli_subsamp
#SBATCH --gres=gpu:2080:1
#SBATCH --ntasks=1
#SBATCH --array=1-5

source activate al_datasets

for ALPHA in 0 25 50 75 100
do
    TOKENIZERS_PARALLELISM=false \
    python mnli_subsamp_expt.py \
        -d mnli \
        --algorithm ERM \
        --model distilbert-base-uncased \
        --root_dir ../data \
        --progress_bar \
        --log_dir ../experiments/mnli_subsamp/ \
        --evaluate_all_splits False \
        --val_metric acc_avg \
        --exp_id ss_$SLURM_ARRAY_TASK_ID \
        --patience 3 \
        --alpha $ALPHA \
        --minority contains_negation \
        --majority contains_negation \
        --download
done
