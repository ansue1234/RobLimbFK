# Seq2Seq Attn
# python train_seq2seq.py --epochs 400 \
#                         --batch_size 1024 \
#                         --exp_name ema_0.2\
#                         --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
#                         --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
#                         --seq_len 100 \
#                         --underlying_model LSTM \
#                         --attention true \
#                         --vel true \
#                         --no_time true \
#                         --seed 1 \
#                         --ema 0.2 \
#                         --tag ema

# python train_seq2seq.py --epochs 400 \
#                         --batch_size 1024 \
#                         --exp_name ema_0.5\
#                         --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
#                         --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
#                         --seq_len 100 \
#                         --underlying_model LSTM \
#                         --attention true \
#                         --vel true \
#                         --no_time true \
#                         --seed 1 \
#                         --ema 0.5 \
#                         --tag ema

python train_seq2seq.py --epochs 400 \
                        --batch_size 1024 \
                        --exp_name ultra_wide_ema0.8\
                        --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
                        --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
                        --seq_len 100 \
                        --underlying_model LSTM \
                        --attention true \
                        --vel true \
                        --no_time true \
                        --hidden_size 1024 \
                        --num_layers 1 \
                        --seed 1 \
                        --ema 0.8 \
                        --tag only_pos
                        
python train_seq2seq.py --epochs 400 \
                        --batch_size 1024 \
                        --exp_name ultra_wide_ema0.5\
                        --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
                        --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
                        --seq_len 100 \
                        --underlying_model LSTM \
                        --attention true \
                        --vel true \
                        --no_time true \
                        --hidden_size 1024 \
                        --num_layers 1 \
                        --seed 1 \
                        --ema 0.5 \
                        --tag only_pos

python train_seq2seq.py --epochs 400 \
                        --batch_size 1024 \
                        --exp_name ultra_wide_ema1\
                        --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
                        --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
                        --seq_len 100 \
                        --underlying_model LSTM \
                        --attention true \
                        --vel true \
                        --no_time true \
                        --hidden_size 1024 \
                        --num_layers 1 \
                        --seed 1 \
                        --ema 1 \
                        --tag only_pos

# python train_seq2seq.py --epochs 400 \
#                         --batch_size 1024 \
#                         --exp_name only_pos_ema0.9\
#                         --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
#                         --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
#                         --seq_len 100 \
#                         --underlying_model LSTM \
#                         --attention true \
#                         --vel true \
#                         --no_time true \
#                         --seed 1 \
#                         --ema 0.9 \
#                         --tag only_pos

# python train_seq2seq.py --epochs 400 \
#                         --batch_size 1024 \
#                         --exp_name ema_0.2\
#                         --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
#                         --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
#                         --seq_len 100 \
#                         --underlying_model LSTM \
#                         --vel true \
#                         --no_time true \
#                         --seed 1 \
#                         --ema 0.2 \
#                         --tag ema

# python train_seq2seq.py --epochs 400 \
#                         --batch_size 1024 \
#                         --exp_name ema_0.5\
#                         --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
#                         --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
#                         --seq_len 100 \
#                         --underlying_model LSTM \
#                         --vel true \
#                         --no_time true \
#                         --seed 1 \
#                         --ema 0.5 \
#                         --tag ema

python train_seq2seq.py --epochs 400 \
                        --batch_size 1024 \
                        --exp_name ultra_wide_ema0.8\
                        --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
                        --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
                        --seq_len 100 \
                        --underlying_model LSTM \
                        --hidden_size 1024 \
                        --num_layers 1 \
                        --vel true \
                        --no_time true \
                        --seed 1 \
                        --ema 0.8 \
                        --tag only_pos

python train_seq2seq.py --epochs 400 \
                        --batch_size 1024 \
                        --exp_name ultra_wide_ema0.5\
                        --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
                        --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
                        --seq_len 100 \
                        --underlying_model LSTM \
                        --hidden_size 1024 \
                        --num_layers 1 \
                        --vel true \
                        --no_time true \
                        --seed 1 \
                        --ema 0.5 \
                        --tag only_pos

python train_seq2seq.py --epochs 400 \
                        --batch_size 1024 \
                        --exp_name ultra_wide_ema1\
                        --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
                        --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
                        --seq_len 100 \
                        --underlying_model LSTM \
                        --hidden_size 512 \
                        --num_layers 1 \
                        --vel true \
                        --no_time true \
                        --seed 1 \
                        --ema 1 \
                        --tag only_pos 
# python train_seq2seq.py --epochs 400 \
#                         --batch_size 1024 \
#                         --exp_name only_pos_ema0.9\
#                         --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
#                         --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
#                         --seq_len 100 \
#                         --underlying_model LSTM \
#                         --vel true \
#                         --no_time true \
#                         --seed 1 \
#                         --ema 0.9 \
#                         --tag only_pos