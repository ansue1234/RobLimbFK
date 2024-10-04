python td3_cont_action.py \
  --env-id "LimbEnv-v0" \
  --seed 0 \
  --exp_name "td3_1M_new" \
  --wandb_project_name "soft_limb_rl" \
  --wandb_entity "gsue" \
  --total_timesteps 1000000\
  --learning_starts 10000 \
  --batch_size 4096 \