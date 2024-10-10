
# python train_GPT.py --epochs 400 \
#                     --batch_size 512 \
#                     --exp_name rollout_gpt_sin_pos\
#                     --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
#                     --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
#                     --seq_len 50 \
#                     --predict_len 15 \
#                     --tag gpt_rollout


python train_GPT.py --epochs 400 \
                    --batch_size 512 \
                    --exp_name rollout_gpt_learned_pos\
                    --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
                    --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
                    --seq_len 500 \
                    --predict_len 25 \
                    --sin_pos false \
                    --tag gpt_rollout
# python train_seq2seq.py --epochs 400 \
#                         --batch_size 512 \
#                         --exp_name rollout_seq2seq\
#                         --train_data_path ../ml_data/purple_no_cool_down_train_data.csv \
#                         --test_data_path ../ml_data/purple_no_cool_down_test_data.csv \
#                         --seq_len 50 \
#                         --predict_len 25 \
#                         --underlying_model LSTM \
#                         --vel true \
#                         --seed 1 \
#                         --tag seq_rollout

