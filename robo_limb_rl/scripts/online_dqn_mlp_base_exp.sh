python online_dqn_mlp.py \
       --exp-name "online_500000_base_exp_simple_discrete" \
       --track \
       --env-id "SafeLimbEnv-v0" \
       --wandb-project_name "soft_limb" \
       --wandb-entity "gsue-research" \
       --total-timesteps 500000 \
       --batch-size 4096 \
       --reward-type "base_exp" \
       --env_config_path "../scripts/yaml/safe_limb_env_simple_discrete_base_exp.yml" \
       --save_model \

# python online_dqn_mlp.py \
#        --exp-name "online_s250000_base_exp" \
#        --track \
#        --env-id "SafeLimbEnv-v0" \
#        --wandb-project_name "soft_limb" \
#        --wandb-entity "gsue-research" \
#        --total-timesteps 500000 \
#        --batch-size 4096 \
#        --reward-type "base_exp" \
#        --env_config_path "../scripts/yaml/safe_limb_env_discrete_base_exp.yml" \
#        --save_model \