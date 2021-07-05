for ID in {6..10}
do
    python mnist.py \
        -e $ID \
        -p /home/johnny/al_datasets/experiments/mnist/ \
        -s random \
        --valid_size 100 \
        --total_data 1000 \
        --batch_size 8 \
        --epochs 50 \
        --patience 10 &
done
