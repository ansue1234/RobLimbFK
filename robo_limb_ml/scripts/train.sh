
python train_LSTM.py --epochs 1 \
                     --batch_size 10 \
                     --num_samples 500 \
                     --exp_name testing_pipeline\
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