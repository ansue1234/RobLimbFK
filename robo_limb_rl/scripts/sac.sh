python sac_cont_action.py \
       --exp-name "sac_1M_new" \
       --env-id "LimbEnv-v0" \
       --wandb-project_name "soft_limb_rl" \
       --wandb-entity "gsue" \
       --total-timesteps 1000000 \
       --learning_starts 4096 \
       --batch_size 4096 \