
python train.py --epochs 2000 \
                --batch_size 2048 \
                --data_path ../data/data.csv \
                --num_samples 16384 \
                --exp_name small_data_large_epoch_seed2\
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