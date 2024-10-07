
python train_LSTM.py --epochs 2 \
                     --batch_size 10 \
                     --num_samples 25001 \
                     --exp_name test_rollout\
                     --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
                     --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
                     --seq_len 50 \
                     --predict_len 40 \
                     --vel true \
                     --no_time true \
                     --seed 1 \




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