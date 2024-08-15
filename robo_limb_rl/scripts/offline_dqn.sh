python offline_dqn.py \
       --exp-name offline_dqn_test_1 \
       --track \
       --wandb-project-name soft_limb \
       --total-timesteps 100 \
       --target-network-frequency 10 \
       --train-frequency 1 \
       --batch-size 32 \
       --network-type LSTM \
       --state-path ../safe_rl_data/states_len10_no_action.npy \
       --next-state-path ../safe_rl_data/next_states_no_action_len10.npy \
       --action-path ../safe_rl_data/actions_len10.npy \
       --reward-path ../safe_rl_data/rewards_reg_markovian_len10.npy

