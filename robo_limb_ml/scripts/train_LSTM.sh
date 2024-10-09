
python train_LSTM_from_check.py --epochs 400 \
                     --batch_size 512 \
                     --exp_name rollout_grad_clip_orig\
                     --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
                     --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
                     --seq_len 50 \
                     --predict_len 25 \
                     --vel true \
                     --no_time true \
                     --seed 1 \
                     --model_path ../model_weights/new_weights/LSTM_b512_e400_s-1_len100_no_cool_down_new_no_time_1724805778 \
                     --tag rollout

# python train_LSTM.py --epochs 400 \
#                      --batch_size 512 \
#                      --exp_name rollout_grad_clip_5_layers\
#                      --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
#                      --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
#                      --seq_len 50 \
#                      --predict_len 10 \
#                      --hidden_size 1024 \
#                      --num_layer 5 \
#                      --vel true \
#                      --no_time true \
#                      --seed 1 \
#                      --tag rollout

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