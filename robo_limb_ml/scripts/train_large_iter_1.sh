for i in $(seq 1 40);
do
    python train.py --epochs 1500 \
                --batch_size 2048 \
                --data_path ../data/data.csv \
                --num_samples 16384 \
                --exp_name t_test_rand_small_data_large_epoch_1500_$i \
                --seed $i
done