for ID in {1..5}
do
    python cifar10.py \
        -e $ID \
        -p /home/johnny/al_datasets/experiments/cifar10/ \
        -s uncertainty \
        --valid_size 1000 \
        --total_data 20000 \
        --batch_size 64 \
        --epochs 100 \
        --patience 50 &
done
