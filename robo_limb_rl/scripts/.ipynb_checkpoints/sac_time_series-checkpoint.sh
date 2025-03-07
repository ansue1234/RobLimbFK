python sac_cont_action.py \
       --exp-name "sac_time_series_200k_seq2seq_enc_no_pretrain_no_freeze_200len" \
       --seed 1 \
       --wandb-project_name "soft_limb_rl" \
       --wandb-entity "gsue" \
       --env-id "LimbEnv-v0" \
       --total-timesteps 200000 \
       --buffer_size 5000 \
       --batch_size 32 \
       --learning_starts 10000 \
       --config_path "./yaml/no_dom/limb_env.yml" \
       --head_type "seq2seq_encoder" \
       --no-freeze_head \
       --no-pretrained_model 

# python sac_cont_action.py \
#        --exp-name "sac_time_series_300k_seq2seq_enc_no_pretrain_freeze" \
#        --seed 1 \
#        --wandb-project_name "soft_limb_rl" \
#        --wandb-entity "gsue" \
#        --env-id "LimbEnv-v0" \
#        --total-timesteps 300000 \
#        --buffer_size 700 \
#        --batch_size 16 \
#        --learning_starts 10000 \
#        --config_path "./yaml/no_dom/limb_env_power_vel.yml" \
#        --head_type "seq2seq_encoder" \
#        --freeze_head \
#        --no-pretrained_model