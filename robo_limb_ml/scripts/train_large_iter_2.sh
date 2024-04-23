for i in $(seq 1 40);
do
    python train.py --epochs 200 \
                    --batch_size 2048 \
                    --data_path ../data/data.csv \
                    --num_samples 131072 \
                    --exp_name t_test_rand_large_data_small_epochs_$i \
                    --seed $i
done