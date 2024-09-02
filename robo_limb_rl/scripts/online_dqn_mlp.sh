# regular reward
python online_dqn_mlp.py \
       --exp-name "online_s250000_reg_loss" \
       --track \
       --env-id "SafeLimbEnv-v0" \
       --wandb-project_name "soft_limb" \
       --wandb-entity "gsue-research" \
       --total-timesteps 250000 \
       --reward-type "reg" \
       --env_config_path "../scripts/yaml/safe_limb_env_discrete.yml" \
       --save_model \

python online_dqn_mlp.py \
       --exp-name "online_s250000_reg_loss_simple_discrete" \
       --track \
       --env-id "SafeLimbEnv-v0" \
       --wandb-project_name "soft_limb" \
       --wandb-entity "gsue-research" \
       --total-timesteps 250000 \
       --reward-type "reg" \
       --env_config_path "../scripts/yaml/safe_limb_env_simple_discrete.yml" \
       --save_model \