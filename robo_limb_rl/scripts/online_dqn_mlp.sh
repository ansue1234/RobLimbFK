# regular reward
python online_dqn_mlp.py \
       --exp-name "online_test" \
       --track \
       --env-id "SafeLimbEnv-v0" \
       --wandb-project_name "soft_limb" \
       --wandb-entity "gsue-research" \
       --total-timesteps 100 \
       --batch-size 2 \
       --learning-starts 2 \
       --target-network_frequency 5 \
       --train-frequency 1 \
       --batch-size 512 \
       --reward-type "reg" \
       --env_config_path "../scripts/yaml/safe_limb_env_discrete.yml"

