
python train_LSTM.py --epochs 200 \
                     --batch_size 512 \
                     --exp_name LSTM_Seq_len_50_prob \
                     --seq_len 50 \
                     --seed 2

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