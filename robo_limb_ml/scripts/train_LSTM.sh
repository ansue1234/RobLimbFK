

python train_LSTM.py --epochs 400 \
                     --batch_size 1024 \
                     --exp_name ema_0.8\
                     --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
                     --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
                     --seq_len 100 \
                     --seed 1 \
                     --tag ema_0.8

python train_RNN.py --epochs 400 \
                    --batch_size 1024 \
                    --exp_name ema_0.8 \
                    --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
                    --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
                    --seed 1 \
                    --seq_len 100 \
                    --tag ema_0.8


# python train.py --epochs 200 \
#                 --batch_size 2048 \
#                 --data_path ../data/data.csv \
#                 --num_samples 131072 \
#                 --exp_name large_data_small_epochs \

# python train.py --epochs 200 \
#                 --batch_size 2048 \
#                 --data_path ../data/syn_data.csv \
#                 --num_samples 131072 \
#                 --exp_name syn_data_small_epochs \

# python train.py --epochs 200 \
#                 --batch_size 2048 \
#                 --data_path ../data/data.csv \
#                 --num_samples 16384 \
#                 --exp_name small_data_small_epochs \