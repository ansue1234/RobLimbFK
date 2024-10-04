# Fine tuning just Seq2Seq

python finetuning.py \
        --num_samples 25000 \
        --exp_name finetune_blue_epoch_models_400 \
        --batch_size 1024 \
        --epochs 400 \
        --train_data_path ../ml_data/blue_no_cool_down_train_data.csv \
        --test_data_path ../ml_data/blue_no_cool_down_test_data.csv \
        --model_path ../model_weights/new_weights/SEQ2SEQ_b1024_e400_s-1_len50_bigger_batch_med_epoch_1728018224 \
        --model_type Seq2Seq \
        --tag fine_tuning_blue_epoch_models

python finetuning.py \
        --num_samples 25000 \
        --exp_name finetune_blue_epoch_models_600 \
        --batch_size 1024 \
        --epochs 400 \
        --train_data_path ../ml_data/blue_no_cool_down_train_data.csv \
        --test_data_path ../ml_data/blue_no_cool_down_test_data.csv \
        --model_path ../model_weights/new_weights/SEQ2SEQ_b1024_e600_s-1_len50_bigger_batch_large_epoch_1728022899 \
        --model_type Seq2Seq \
        --tag fine_tuning_blue_epoch_models

# Fine tuning just Seq2Seq attention
python finetuning.py \
        --num_samples 25000 \
        --exp_name finetune_blue_epoch_models_400 \
        --batch_size 1024 \
        --epochs 400 \
        --train_data_path ../ml_data/blue_no_cool_down_train_data.csv \
        --test_data_path ../ml_data/blue_no_cool_down_test_data.csv \
        --model_path ../model_weights/new_weights/SEQ2SEQ_ATTENTION_b1024_e400_s-1_len50_bigger_batch_med_epoch_1728016729 \
        --model_type Seq2Seq \
        --attention true \
        --tag fine_tuning_blue_epoch_models

python finetuning.py \
        --num_samples 25000 \
        --exp_name finetune_blue_epoch_models_600 \
        --batch_size 1024 \
        --epochs 400 \
        --train_data_path ../ml_data/blue_no_cool_down_train_data.csv \
        --test_data_path ../ml_data/blue_no_cool_down_test_data.csv \
        --model_path ../model_weights/new_weights/SEQ2SEQ_ATTENTION_b1024_e600_s-1_len50_bigger_batch_large_epoch_1728020659 \
        --model_type Seq2Seq \
        --attention true \
        --tag fine_tuning_blue_epoch_models

# LSTM
python finetuning.py \
        --num_samples 25000 \
        --exp_name finetune_blue_epoch_models_400 \
        --batch_size 1024 \
        --epochs 400 \
        --train_data_path ../ml_data/blue_no_cool_down_train_data.csv \
        --test_data_path ../ml_data/blue_no_cool_down_test_data.csv \
        --model_path ../model_weights/new_weights/LSTM_b1024_e400_s-1_len50_bigger_batch_med_epoch_1728019254 \
        --model_type LSTM \
        --tag fine_tuning_blue_epoch_models

python finetuning.py \
        --num_samples 25000 \
        --exp_name finetune_blue_epoch_models_600 \
        --batch_size 1024 \
        --epochs 400 \
        --train_data_path ../ml_data/blue_no_cool_down_train_data.csv \
        --test_data_path ../ml_data/blue_no_cool_down_test_data.csv \
        --model_path ../model_weights/new_weights/LSTM_b1024_e600_s-1_len50_bigger_batch_large_epoch_1728024441 \
        --model_type LSTM \
        --tag fine_tuning_blue_epoch_models