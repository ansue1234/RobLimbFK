python sac_mlp.py \
       --exp-name "sac_250000_reg" \
       --track \
       --env-id "SafeLimbEnv-v0" \
       --wandb-project_name "soft_limb" \
       --wandb-entity "gsue-research" \
       --total-timesteps 250000 \
       --buffer_size 10000 \
       --learning_starts 4096 \
       --batch_size 4096 \
       --env_config_path "../scripts/yaml/safe_limb_env.yml" \

python sac_mlp.py \
       --exp-name "sac_250000_base_exp" \
       --track \
       --env-id "SafeLimbEnv-v0" \
       --wandb-project_name "soft_limb" \
       --wandb-entity "gsue-research" \
       --total-timesteps 250000 \
       --buffer_size 10000 \
       --learning_starts 4096 \
       --batch_size 4096 \
       --env_config_path "../scripts/yaml/safe_limb_env_base_exp.yml" \