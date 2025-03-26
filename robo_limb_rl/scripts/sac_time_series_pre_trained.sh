# python sac_cont_action_time_series.py \
#        --exp-name "sac_no_freeze_no_pretrain_100k_power_encoder" \
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
#        --head_type "seq2seq_encoder" \
#        --no-pretrained-model \
#        --no-freeze-head \

# python sac_cont_action_time_series.py \
#        --exp-name "sac_freeze_pretrain_100k_no_power_dyn_goal_encoder" \
#        --seed 1 \
#        --wandb-project_name "soft_limb_rl" \
#        --wandb-entity "gsue" \
#        --env-id "LimbEnv-v0" \
#        --total-timesteps 100000 \
#        --buffer_size 1000000 \
#        --batch_size 256 \
#        --learning_starts 1000 \
#        --num_envs 10 \
#        --config_path "./yaml/no_dom/limb_env_no_power_vel.yml" \
#        --head_type "seq2seq_encoder" \
#        --pretrained_model \
#        --freeze_head \
#        --adjust_goal_tolerance\

# python sac_cont_action_time_series.py \
#        --exp-name "sac_freeze_pretrain_100k_no_power_no_dyn_goal_encoder" \
#        --seed 1 \
#        --wandb-project_name "soft_limb_rl" \
#        --wandb-entity "gsue" \
#        --env-id "LimbEnv-v0" \
#        --total-timesteps 100000 \
#        --buffer_size 1000000 \
#        --batch_size 256 \
#        --learning_starts 1000 \
#        --num_envs 10 \
#        --config_path "./yaml/no_dom/limb_env_no_power_vel.yml" \
#        --head_type "seq2seq_encoder" \
#        --pretrained_model \
#        --freeze_head \
#        --no-adjust-goal-tolerance \

python sac_cont_action_time_series.py \
       --exp-name "sac_freeze_pretrain_100k_power_dyn_goal_encoder_real" \
       --seed 1 \
       --wandb-project_name "soft_limb_rl" \
       --wandb-entity "gsue" \
       --env-id "LimbEnv-v0" \
       --total-timesteps 100000 \
       --buffer_size 1000000 \
       --batch_size 256 \
       --learning_starts 1000 \
       --num_envs 10 \
       --config_path "./yaml/no_dom/limb_env_power_vel.yml"\
       --head_type "seq2seq_encoder" \
       --pretrained_model \
       --freeze_head \
       # --adjust-goal-tolerance \
# python sac_cont_action_time_series.py \
#        --exp-name "sac_no_freeze_pretrain_100k_power_encoder" \
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
#        --head_type "seq2seq_encoder" \
#        --pretrained-model \
#        --no-freeze-head \

# python sac_cont_action_time_series.py \
#        --exp-name "sac_freeze_no_pretrain_100k_power_encoder" \
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
#        --head_type "seq2seq_encoder" \
#        --no-pretrained-model \
#        --freeze-head \






# python sac_cont_action.py \
#        --exp-name "sac_pos_rew_600k_pretrain_no_freeze_200len_512" \
#        --seed 1 \
#        --wandb-project_name "soft_limb_rl" \
#        --wandb-entity "gsue" \
#        --env-id "LimbEnv-v0" \
#        --total-timesteps 600000 \
#        --buffer_size 10000 \
#        --batch_size 32 \
#        --learning_starts 10000 \
#        --config_path "./yaml/no_dom/limb_env_no_power_vel.yml" \
#        --head_type "seq2seq_encoder" \
#        --pretrained_model \
#        --no-freeze_head \

# python sac_cont_action.py \
#        --exp-name "sac_pos_rew_600k_no_pretrain_no_freeze_200len_512" \
#        --seed 1 \
#        --wandb-project_name "soft_limb_rl" \
#        --wandb-entity "gsue" \
#        --env-id "LimbEnv-v0" \
#        --total-timesteps 600000 \
#        --buffer_size 10000 \
#        --batch_size 32 \
#        --learning_starts 10000 \
#        --config_path "./yaml/no_dom/limb_env_no_power_vel.yml" \
#        --head_type "seq2seq_encoder" \
#        --no_pretrained_model \
#        --no_freeze_head \

# python sac_cont_action.py \
#        --exp-name "sac_pos_rew_600k_no_pretrain_freeze_200len_512" \
#        --seed 1 \
#        --wandb-project_name "soft_limb_rl" \
#        --wandb-entity "gsue" \
#        --env-id "LimbEnv-v0" \
#        --total-timesteps 600000 \
#        --buffer_size 10000 \
#        --batch_size 32 \
#        --learning_starts 10000 \
#        --config_path "./yaml/no_dom/limb_env_no_power_vel.yml" \
#        --head_type "seq2seq_encoder" \
#        --no_pretrained_model \
#        --freeze_head \
# python sac_cont_action.py \
#        --exp-name "sac_pos_reward_200k_seq2seq_enc_no_pretrain_no_freeze_200len_norm" \
#        --seed 1 \
#        --wandb-project_name "soft_limb_rl" \
#        --wandb-entity "gsue" \
#        --env-id "LimbEnv-v0" \
#        --total-timesteps 200000 \
#        --buffer_size 5000 \
#        --batch_size 32 \
#        --learning_starts 10000 \
#        --config_path "./yaml/no_dom/limb_env_no_power_vel.yml" \
#        --head_type "seq2seq_encoder" \
#        --no-pretrained_model \
#        --no-freeze_head \

# python sac_cont_action.py \
#        --exp-name "sac_pos_rew_600k_pretrain_freeze_200len_512" \
#        --seed 1 \
#        --wandb-project_name "soft_limb_rl" \
#        --wandb-entity "gsue" \
#        --env-id "LimbEnv-v0" \
#        --total-timesteps 600000 \
#        --buffer_size 10000 \
#        --batch_size 32 \
#        --learning_starts 10000 \
#        --config_path "./yaml/no_dom/limb_env_no_power_vel.yml" \
#        --head_type "seq2seq_encoder" \
#        --pretrained_model \
#        --freeze_head \

# # python sac_cont_action.py \
# #        --exp-name "sac_pos_rew_600k_pretrain_no_freeze_200len_512" \
# #        --seed 1 \
# #        --wandb-project_name "soft_limb_rl" \
# #        --wandb-entity "gsue" \
# #        --env-id "LimbEnv-v0" \
# #        --total-timesteps 600000 \
# #        --buffer_size 10000 \
# #        --batch_size 32 \
# #        --learning_starts 10000 \
# #        --config_path "./yaml/no_dom/limb_env_no_power_vel.yml" \
# #        --head_type "seq2seq_encoder" \
# #        --pretrained_model \
# #        --no-freeze_head \

# # python sac_cont_action.py \
# #        --exp-name "sac_pos_rew_600k_no_pretrain_no_freeze_200len_512" \
# #        --seed 1 \
# #        --wandb-project_name "soft_limb_rl" \
# #        --wandb-entity "gsue" \
# #        --env-id "LimbEnv-v0" \
# #        --total-timesteps 600000 \
# #        --buffer_size 10000 \
# #        --batch_size 32 \
# #        --learning_starts 10000 \
# #        --config_path "./yaml/no_dom/limb_env_no_power_vel.yml" \
# #        --head_type "seq2seq_encoder" \
# #        --no_pretrained_model \
# #        --no_freeze_head \

# # python sac_cont_action.py \
# #        --exp-name "sac_pos_rew_600k_no_pretrain_freeze_200len_512" \
# #        --seed 1 \
# #        --wandb-project_name "soft_limb_rl" \
# #        --wandb-entity "gsue" \
# #        --env-id "LimbEnv-v0" \
# #        --total-timesteps 600000 \
# #        --buffer_size 10000 \
# #        --batch_size 32 \
# #        --learning_starts 10000 \
# #        --config_path "./yaml/no_dom/limb_env_no_power_vel.yml" \
# #        --head_type "seq2seq_encoder" \
# #        --no_pretrained_model \
# #        --freeze_head \
# # python sac_cont_action.py \
# #        --exp-name "sac_pos_reward_200k_seq2seq_enc_no_pretrain_no_freeze_200len_norm" \
# #        --seed 1 \
# #        --wandb-project_name "soft_limb_rl" \
# #        --wandb-entity "gsue" \
# #        --env-id "LimbEnv-v0" \
# #        --total-timesteps 200000 \
# #        --buffer_size 5000 \
# #        --batch_size 32 \
# #        --learning_starts 10000 \
# #        --config_path "./yaml/no_dom/limb_env_no_power_vel.yml" \
# #        --head_type "seq2seq_encoder" \
# #        --no-pretrained_model \
# #        --no-freeze_head \