# regular reward
python online_dqn_mlp.py \
       --exp-name "online_first_attempt_s500000" \
       --track \
       --env-id "SafeLimbEnv-v0" \
       --wandb-project_name "soft_limb" \
       --wandb-entity "gsue-research" \
       --total-timesteps 500000 \
       --reward-type "reg" \
       --env_config_path "../scripts/yaml/safe_limb_env_discrete.yml" \
       --save_model \
