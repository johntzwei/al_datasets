source activate al_datasets

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false \
python wilds_al_expt.py \
    -d mnli \
    --algorithm ERM \
    --model distilbert-base-uncased \
    --root_dir ../data \
    --progress_bar \
    --log_dir ../experiments/mnli/ \
    --evaluate_all_splits False \
    --al_strategy us \
    --total_data 20000 \
    --valid_size 2000 \
    --val_metric acc_avg \
    --patience 3 \
    --exp_id us_$SLURM_ARRAY_TASK_ID \
    --rounds 10 \
    --download

