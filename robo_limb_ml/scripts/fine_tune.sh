# Fine tuning just Seq2Seq

python finetuning.py \
        --num_samples 25000 \
        --exp_name len50 \
        --batch_size 512 \
        --epochs 200 \
        --train_data_path ../ml_data/purple_train_data.csv \
        --test_data_path ../ml_data/purple_test_data.csv \
        --model_path ../model_weights/new_weights/SEQ2SEQ_b512_e200_s-1_Seq2Seq_len50_LSTM_1722973641 \
        --model_type Seq2Seq \
        --tag fine_tuning_e200

python finetuning.py \
        --num_samples 50000 \
        --batch_size 512 \
        --exp_name len50 \
        --epochs 200 \
        --train_data_path ../ml_data/purple_train_data.csv \
        --test_data_path ../ml_data/purple_test_data.csv \
        --model_path ../model_weights/new_weights/SEQ2SEQ_b512_e200_s-1_Seq2Seq_len50_LSTM_1722973641 \
        --model_type Seq2Seq \
        --tag fine_tuning_e200

python finetuning.py \
        --num_samples 75000 \
        --batch_size 512 \
        --exp_name len50 \
        --epochs 200 \
        --train_data_path ../ml_data/purple_train_data.csv \
        --test_data_path ../ml_data/purple_test_data.csv \
        --model_path ../model_weights/new_weights/SEQ2SEQ_b512_e200_s-1_Seq2Seq_len50_LSTM_1722973641 \
        --model_type Seq2Seq \
        --tag fine_tuning_e200

python finetuning.py \
        --batch_size 512 \
        --epochs 200 \
        --exp_name len50 \
        --train_data_path ../ml_data/purple_train_data.csv \
        --test_data_path ../ml_data/purple_test_data.csv \
        --model_path ../model_weights/new_weights/SEQ2SEQ_b512_e200_s-1_Seq2Seq_len50_LSTM_1722973641 \
        --model_type Seq2Seq \
        --tag fine_tuning_e200

# Fine tuning just Seq2Seq
python finetuning.py \
        --num_samples 25000 \
        --batch_size 512 \
        --epochs 200 \
        --exp_name len50 \
        --train_data_path ../ml_data/purple_train_data.csv \
        --test_data_path ../ml_data/purple_test_data.csv \
        --model_path ../model_weights/new_weights/SEQ2SEQ_ATTENTION_b512_e200_s-1_Seq2Seq_len50_LSTM_1722928107 \
        --model_type Seq2Seq \
        --attention true
        --tag fine_tuning_e200

python finetuning.py \
        --num_samples 50000 \
        --batch_size 512 \
        --epochs 200 \
        --exp_name len50 \
        --train_data_path ../ml_data/purple_train_data.csv \
        --test_data_path ../ml_data/purple_test_data.csv \
        --model_path ../model_weights/new_weights/SEQ2SEQ_ATTENTION_b512_e200_s-1_Seq2Seq_len50_LSTM_1722928107 \
        --model_type Seq2Seq \
        --attention true
        --tag fine_tuning_e200

python finetuning.py \
        --num_samples 75000 \
        --batch_size 512 \
        --epochs 200 \
        --exp_name len50 \
        --train_data_path ../ml_data/purple_train_data.csv \
        --test_data_path ../ml_data/purple_test_data.csv \
        --model_path ../model_weights/new_weights/SEQ2SEQ_ATTENTION_b512_e200_s-1_Seq2Seq_len50_LSTM_1722928107 \
        --model_type Seq2Seq \
        --attention true \
        --tag fine_tuning_e200

python finetuning.py \
        --batch_size 512 \
        --epochs 200 \
        --exp_name len50 \
        --train_data_path ../ml_data/purple_train_data.csv \
        --test_data_path ../ml_data/purple_test_data.csv \
        --model_path ../model_weights/new_weights/SEQ2SEQ_ATTENTION_b512_e200_s-1_Seq2Seq_len50_LSTM_1722928107 \
        --model_type Seq2Seq \
        --attention true \
        --tag fine_tuning_e200

# LSTM
python finetuning.py \
        --num_samples 25000 \
        --exp_name len50 \
        --batch_size 512 \
        --epochs 200 \
        --train_data_path ../ml_data/purple_train_data.csv \
        --test_data_path ../ml_data/purple_test_data.csv \
        --model_path ../model_weights/new_weights/LSTM_b512_e200_s-1_LSTM_len50_1722938560 \
        --model_type LSTM \
        --tag fine_tuning_e200

python finetuning.py \
        --num_samples 50000 \
        --exp_name len50 \
        --batch_size 512 \
        --epochs 200 \
        --train_data_path ../ml_data/purple_train_data.csv \
        --test_data_path ../ml_data/purple_test_data.csv \
        --model_path ../model_weights/new_weights/LSTM_b512_e200_s-1_LSTM_len50_1722938560 \
        --model_type LSTM \
        --tag fine_tuning_e200

python finetuning.py \
        --num_samples 75000 \
        --exp_name len50 \
        --batch_size 512 \
        --epochs 200 \
        --train_data_path ../ml_data/purple_train_data.csv \
        --test_data_path ../ml_data/purple_test_data.csv \
        --model_path ../model_weights/new_weights/LSTM_b512_e200_s-1_LSTM_len50_1722938560 \
        --model_type LSTM \
        --tag fine_tuning_e200

python finetuning.py \
        --exp_name len50 \
        --batch_size 512 \
        --epochs 200 \
        --train_data_path ../ml_data/purple_train_data.csv \
        --test_data_path ../ml_data/purple_test_data.csv \
        --model_path ../model_weights/new_weights/LSTM_b512_e200_s-1_LSTM_len50_1722938560 \
        --model_type LSTM \
        --tag fine_tuning_e200