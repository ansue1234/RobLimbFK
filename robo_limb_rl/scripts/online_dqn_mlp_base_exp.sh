python online_dqn_mlp.py \
       --exp-name "online_s500000_base_exp_loss_1_layer_512" \
       --track \
       --env-id "SafeLimbEnv-v0" \
       --wandb-project_name "soft_limb" \
       --wandb-entity "gsue-research" \
       --total-timesteps 200000 \
       --reward-type "base_exp" \
       --env_config_path "../scripts/yaml/safe_limb_env_discrete_base_exp.yml" \
       --save_model \