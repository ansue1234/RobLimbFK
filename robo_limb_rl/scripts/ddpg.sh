python ddpg_cont_action.py \
  --env-id "LimbEnv-v0" \
  --exp_name "ddpg_cont_act_250k" \
  --wandb_project_name "soft_limb_rl" \
  --wandb_entity "gsue" \
  --total_timesteps 250000 \
  --learning_starts 10000 \
  --batch_size 4096 \