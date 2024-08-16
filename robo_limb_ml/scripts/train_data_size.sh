# Seq2Seq no 
# python train_seq2seq.py --epochs 400 \
#                         --batch_size 512 \
#                         --exp_name Seq2Seq_len50_LSTM_20000 \
#                         --num_samples 20000 \
#                         --seq_len 50 \
#                         --underlying_model LSTM \
#                         --seed 1

python train_seq2seq.py --epochs 600 \
                        --batch_size 512 \
                        --exp_name Seq2Seq_len50_LSTM_50000_e600 \
                        --num_samples 50000 \
                        --seq_len 50 \
                        --underlying_model LSTM \
                        --seed 1

# python train_seq2seq.py --epochs 400 \
#                         --batch_size 512 \
#                         --exp_name Seq2Seq_len50_LSTM_75000 \
#                         --num_samples 75000 \
#                         --seq_len 50 \
#                         --underlying_model LSTM \
#                         --seed 1

# python train_seq2seq.py --epochs 200 \
#                         --batch_size 512 \
#                         --exp_name Seq2Seq_Attn_len50_LSTM \
#                         --num_samples 20000 \
#                         --seq_len 50 \
#                         --underlying_model LSTM \
#                         --attention true\
#                         --seed 1

# python train_seq2seq.py --epochs 200 \
#                         --batch_size 512 \
#                         --exp_name Seq2Seq_Attn_len50_LSTM \
#                         --num_samples 50000 \
#                         --seq_len 50 \
#                         --underlying_model LSTM \
#                         --attention true\
#                         --seed 1

# python train_seq2seq.py --epochs 200 \
#                         --batch_size 512 \
#                         --exp_name Seq2Seq_Attn_len50_LSTM \
#                         --num_samples 75000 \
#                         --seq_len 50 \
#                         --attention true\
#                         --underlying_model LSTM \
#                         --seed 1
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
#                     --exp_name RNN_len50\
#                     --seq_len 50 \
#                     --seed 1

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
#                      --exp_name LSTM_len50\
#                      --seq_len 50 \
#                      --seed 1

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