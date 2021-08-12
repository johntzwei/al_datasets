source activate al_datasets

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false \
python mnli_subsamp_expt.py \
    -d civilcomments \
    --algorithm ERM \
    --model distilbert-base-uncased \
    --root_dir ../data \
    --progress_bar \
    --log_dir ../experiments/civilcomments_subsamp/ \
    --evaluate_all_splits False \
    --val_metric acc_avg \
    --exp_id ss_test \
    --patience 3 \
    --alpha 0 \
    --minority contains_negation \
    --majority contains_negation \
    --download
