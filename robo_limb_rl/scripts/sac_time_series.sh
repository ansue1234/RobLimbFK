python sac_cont_action_time_series.py \
       --exp-name "sac_freeze_pretrain_100k_encoder_long_seq" \
       --seed 1 \
       --wandb-project_name "soft_limb_rl" \
       --wandb-entity "gsue" \
       --env-id "LimbEnv-v0" \
       --total-timesteps 100000 \
       --buffer_size 1000000 \
       --batch_size 256 \
       --learning_starts 1000 \
       --num_envs 10 \
       --config_path "./yaml/no_dom/limb_env_power_vel.yml" \
       --head_type "seq2seq_encoder" \
       --seq_len 1600 \
       --pretrained-model \
       --freeze-head \

# python sac_cont_action_time_series.py \
#        --exp-name "sac_freeze_pretrain_100k_power_full" \
#        --seed 1 \
#        --wandb-project_name "soft_limb_rl" \
#        --wandb-entity "gsue" \
#        --env-id "LimbEnv-v0" \
#        --total-timesteps 100000 \
#        --buffer_size 1000000 \
#        --batch_size 256 \
#        --learning_starts 1000 \
#        --num_envs 10 \
#        --config_path "./yaml/no_dom/limb_env_power_vel.yml" \
#        --head_type "seq2seq_full" \
#        --pretrained_model \
#        --freeze_head \

# python sac_cont_action_time_series.py \
#        --exp-name "sac_no_freeze_pretrain_100k_power_full" \
#        --seed 1 \
#        --wandb-project_name "soft_limb_rl" \
#        --wandb-entity "gsue" \
#        --env-id "LimbEnv-v0" \
#        --total-timesteps 100000 \
#        --buffer_size 1000000 \
#        --batch_size 256 \
#        --learning_starts 1000 \
#        --num_envs 10 \
#        --config_path "./yaml/no_dom/limb_env_power_vel.yml" \
#        --head_type "seq2seq_full" \
#        --pretrained-model \
#        --no-freeze-head \

# python sac_cont_action_time_series.py \
#        --exp-name "sac_freeze_no_pretrain_100k_power_full" \
#        --seed 1 \
#        --wandb-project_name "soft_limb_rl" \
#        --wandb-entity "gsue" \
#        --env-id "LimbEnv-v0" \
#        --total-timesteps 100000 \
#        --buffer_size 1000000 \
#        --batch_size 256 \
#        --learning_starts 1000 \
#        --num_envs 10 \
#        --config_path "./yaml/no_dom/limb_env_power_vel.yml" \
#        --head_type "seq2seq_full" \
#        --no-pretrained-model \
#        --freeze-head \