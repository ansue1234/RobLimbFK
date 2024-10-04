# Seq2Seq Attn
python train_seq2seq.py --epochs 400 \
                        --batch_size 1024 \
                        --num_samples 25000 \
                        --exp_name blue_no_cool_down_25k\
                        --train_data_path ../ml_data/blue_no_cool_down_train_data.csv \
                        --test_data_path ../ml_data/blue_no_cool_down_test_data.csv \
                        --seq_len 50 \
                        --underlying_model LSTM \
                        --attention true \
                        --vel true \
                        --no_time true \
                        --seed 1 \
                        --tag blue_no_cool_down

python train_seq2seq.py --epochs 400 \
                        --num_samples 25000 \
                        --batch_size 1024 \
                        --exp_name blue_no_cool_down_25k\
                        --train_data_path ../ml_data/blue_no_cool_down_train_data.csv \
                        --test_data_path ../ml_data/blue_no_cool_down_test_data.csv \
                        --seq_len 50 \
                        --underlying_model LSTM \
                        --vel true \
                        --no_time true \
                        --seed 1 \
                        --tag blue_no_cool_down

python train_LSTM.py --epochs 400 \
                     --num_samples 25000 \
                     --batch_size 1024 \
                     --exp_name blue_no_cool_down_25k\
                     --train_data_path ../ml_data/blue_no_cool_down_train_data.csv \
                     --test_data_path ../ml_data/blue_no_cool_down_test_data.csv \
                     --seq_len 50 \
                     --vel true \
                     --no_time true \
                     --seed 1 \
                     --tag blue_no_cool_down

python train_RNN.py --epochs 400 \
                    --batch_size 1024 \
                    --num_samples 25000 \
                    --exp_name blue_no_cool_down_25k\
                    --train_data_path ../ml_data/blue_no_cool_down_train_data.csv \
                    --test_data_path ../ml_data/blue_no_cool_down_test_data.csv \
                    --seq_len 50 \
                    --vel true \
                    --no_time true \
                    --seed 1 \
                    --tag blue_no_cool_down

# 600 epochs
# python train_seq2seq.py --epochs 600 \
#                         --batch_size 1024 \
#                         --exp_name bigger_batch_large_epoch\
#                         --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
#                         --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
#                         --seq_len 50 \
#                         --underlying_model LSTM \
#                         --attention true \
#                         --vel true \
#                         --no_time true \
#                         --seed 1 \
#                         --tag seq_50_no_time_no_cool_down

# python train_seq2seq.py --epochs 600 \
#                         --batch_size 1024 \
#                         --exp_name bigger_batch_large_epoch\
#                         --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
#                         --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
#                         --seq_len 50 \
#                         --underlying_model LSTM \
#                         --vel true \
#                         --no_time true \
#                         --seed 1 \
#                         --tag seq_50_no_time_no_cool_down

# python train_LSTM.py --epochs 600 \
#                      --batch_size 1024 \
#                      --exp_name bigger_batch_large_epoch\
#                      --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
#                      --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
#                      --seq_len 50 \
#                      --vel true \
#                      --no_time true \
#                      --seed 1 \
#                      --tag seq_50_no_time_no_cool_down

# python train_RNN.py --epochs 600 \
#                     --batch_size 1024 \
#                     --exp_name bigger_batch_large_epoch\
#                     --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
#                     --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
#                     --seq_len 50 \
#                     --vel true \
#                     --no_time true \
#                     --seed 1 \
#                     --tag seq_50_no_time_no_cool_down

# python train_seq2seq.py --epochs 200 \
#                         --batch_size 512 \
#                         --exp_name Seq2Seq_len10_RNN \
#                         --seq_len 10 \
#                         --underlying_model RNN \
#                         --attention true \
#                         --seed 1

# python train_seq2seq.py --epochs 200 \
#                         --batch_size 512 \
#                         --exp_name Seq2Seq_len50_RNN \
#                         --seq_len 50 \
#                         --underlying_model RNN \
#                         --attention true \
#                         --seed 1

# Seq2Seq no 
# python train_seq2seq.py --epochs 200 \
#                         --batch_size 512 \
#                         --exp_name Seq2Seq_len10_LSTM \
#                         --seq_len 10 \
#                         --underlying_model LSTM \
#                         --seed 1

# python train_seq2seq.py --epochs 200 \
#                         --batch_size 512 \
#                         --exp_name vel\
#                         --train_data_path ../ml_data/train_data.csv \
#                         --test_data_path ../ml_data/test_data.csv \
#                         --seq_len 50 \
#                         --underlying_model LSTM \
#                         --vel true \
#                         --seed 1 \
#                         --tag vel


# python train_seq2seq.py --epochs 200 \
#                         --batch_size 512 \
#                         --exp_name Seq2Seq_len10_RNN \
#                         --seq_len 10 \
#                         --underlying_model RNN \
#                         --attention false \
#                         --seed 1

# python train_seq2seq.py --epochs 200 \
#                         --batch_size 512 \
#                         --exp_name Seq2Seq_len50_RNN \
#                         --seq_len 50 \
#                         --underlying_model RNN \
#                         --attention false \
#                         --seed 1

# # RNN
# python train_RNN.py --epochs 200 \
#                     --batch_size 512 \
#                     --exp_name vel\
#                     --seq_len 50 \
#                     --vel true \
#                     --seed 1 \
#                     --tag vel

# python train_RNN.py --epochs 200 \
#                     --batch_size 512 \
#                     --exp_name RNN_len10\
#                     --seq_len 10 \
#                     --seed 1

# # MLP
# python train_MLP.py --epochs 200 \
#                     --batch_size 512 \
#                     --exp_name MLP_len50\
#                     --seq_len 50 \
#                     --seed 1

# python train_MLP.py --epochs 200 \
#                     --batch_size 512 \
#                     --exp_name MLP_len10\
#                     --seq_len 10 \
#                     --seed 1

# # LSTM Stateful
# python train_LSTM.py --epochs 200 \
#                      --batch_size 512 \
#                      --exp_name vel\
#                      --train_data_path ../ml_data/train_data.csv \
#                      --test_data_path ../ml_data/test_data.csv \
#                      --seq_len 50 \
#                      --vel true \
#                      --seed 1 \
#                      --tag vel

# python train_LSTM.py --epochs 200 \
#                      --batch_size 512 \
#                      --exp_name LSTM_len10\
#                      --seq_len 10 \
#                      --seed 1

# # LSTM Stateless
# python train_LSTM.py --epochs 200 \
#                      --batch_size 512 \
#                      --exp_name LSTM_len50_stateless\
#                      --seq_len 50 \
#                      --state stateless \
#                      --seed 1

# python train_LSTM.py --epochs 200 \
#                      --batch_size 512 \
#                      --exp_name LSTM_len10_stateless\
#                      --seq_len 10 \
#                      --state stateless \
#                      --seed 1

# #RNN stateless
# # RNN
# python train_RNN.py --epochs 200 \
#                     --batch_size 512 \
#                     --exp_name RNN_len50_stateless\
#                     --seq_len 50 \
#                     --state stateless \
#                     --seed 1

# python train_RNN.py --epochs 200 \
#                     --batch_size 512 \
#                     --exp_name RNN_len10_stateless\
#                     --seq_len 10 \
#                     --state stateless \
#                     --seed 1