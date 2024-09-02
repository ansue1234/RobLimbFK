# regular reward
python online_dqn_lstm.py \
       --exp-name "lstm_100000" \
       --env-id "SafeLimbEnv-v0" \
       --track \
       --wandb-project_name "soft_limb" \
       --wandb-entity "gsue-research" \
       --total-timesteps 100000 \
       --buffer_size 10000 \
       --batch-size 4096 \
       --learning_starts 4096 \
       --reward-type "reg" \
       --env_config_path "../scripts/yaml/safe_limb_env_simple_discrete.yml" \
       --seq_len 50 \
       --save_model \

python online_dqn_lstm.py \
       --exp-name "lstm_100000_base_exp" \
       --env-id "SafeLimbEnv-v0" \
       --track \
       --wandb-project_name "soft_limb" \
       --wandb-entity "gsue-research" \
       --total-timesteps 100000 \
       --buffer_size 10000 \
       --batch-size 4096 \
       --learning_starts 4096 \
       --reward-type "base_exp" \
       --env_config_path "../scripts/yaml/safe_limb_env_simple_discrete_base_exp.yml" \
       --seq_len 50 \
       --save_model \

