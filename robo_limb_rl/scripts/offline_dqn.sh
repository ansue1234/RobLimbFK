# regular reward
python offline_dqn.py \
       --exp-name offline_dqn_reg_no_act_no_shuffle_500000 \
       --track \
       --wandb-project-name soft_limb \
       --total-timesteps 500000 \
       --target-network-frequency 500 \
       --train-frequency 10 \
       --batch-size 512 \
       --network-type LSTM \
       --state-path ../safe_rl_data/states_len10_no_action.npy \
       --next-state-path ../safe_rl_data/next_states_no_action_len10.npy \
       --action-path ../safe_rl_data/actions_len10.npy \
       --reward-path ../safe_rl_data/rewards_reg_markovian_len10.npy \
       --reward_type reg

python offline_dqn.py \
       --exp-name offline_dqn_reg_no_act_shuffle_500000 \
       --track \
       --wandb-project-name soft_limb \
       --total-timesteps 500000 \
       --target-network-frequency 500 \
       --train-frequency 10 \
       --batch-size 512 \
       --network-type LSTM \
       --state-path ../safe_rl_data/states_len10_no_action.npy \
       --next-state-path ../safe_rl_data/next_states_no_action_len10.npy \
       --action-path ../safe_rl_data/actions_len10.npy \
       --reward-path ../safe_rl_data/rewards_reg_markovian_len10.npy \
       --reward_type reg \
       --shuffle

python offline_dqn.py \
       --exp-name offline_dqn_reg_shuffle_500000 \
       --track \
       --wandb-project-name soft_limb \
       --total-timesteps 500000 \
       --target-network-frequency 500 \
       --train-frequency 10 \
       --batch-size 512 \
       --network-type LSTM \
       --state-path ../safe_rl_data/states_len10.npy \
       --next-state-path ../safe_rl_data/next_states_len10.npy \
       --action-path ../safe_rl_data/actions_len10.npy \
       --reward-path ../safe_rl_data/rewards_reg_markovian_len10.npy \
       --reward_type reg \
       --shuffle

python offline_dqn.py \
       --exp-name offline_dqn_reg_no_shuffle_500000 \
       --track \
       --wandb-project-name soft_limb \
       --total-timesteps 500000 \
       --target-network-frequency 500 \
       --train-frequency 10 \
       --batch-size 512 \
       --network-type LSTM \
       --state-path ../safe_rl_data/states_len10.npy \
       --next-state-path ../safe_rl_data/next_states_len10.npy \
       --action-path ../safe_rl_data/actions_len10.npy \
       --reward-path ../safe_rl_data/rewards_reg_markovian_len10.npy \
       --reward_type reg

# base e exp reward
python offline_dqn.py \
       --exp-name offline_dqn_e_exp_no_act_no_shuffle_500000 \
       --track \
       --wandb-project-name soft_limb \
       --total-timesteps 500000 \
       --target-network-frequency 500 \
       --train-frequency 10 \
       --batch-size 512 \
       --network-type LSTM \
       --state-path ../safe_rl_data/states_len10_no_action.npy \
       --next-state-path ../safe_rl_data/next_states_no_action_len10.npy \
       --action-path ../safe_rl_data/actions_len10.npy \
       --reward-path ../safe_rl_data/rewards_base_e_exponential_len10.npy \
       --reward_type e_exp

python offline_dqn.py \
       --exp-name offline_dqn_e_exp_no_act_shuffle_500000 \
       --track \
       --wandb-project-name soft_limb \
       --total-timesteps 500000 \
       --target-network-frequency 500 \
       --train-frequency 10 \
       --batch-size 512 \
       --network-type LSTM \
       --state-path ../safe_rl_data/states_len10_no_action.npy \
       --next-state-path ../safe_rl_data/next_states_no_action_len10.npy \
       --action-path ../safe_rl_data/actions_len10.npy \
       --reward-path ../safe_rl_data/rewards_base_e_exponential_len10.npy \
       --reward_type e_exp \
       --shuffle

python offline_dqn.py \
       --exp-name offline_dqn_e_exp_shuffle_500000 \
       --track \
       --wandb-project-name soft_limb \
       --total-timesteps 500000 \
       --target-network-frequency 500 \
       --train-frequency 10 \
       --batch-size 512 \
       --network-type LSTM \
       --state-path ../safe_rl_data/states_len10.npy \
       --next-state-path ../safe_rl_data/next_states_len10.npy \
       --action-path ../safe_rl_data/actions_len10.npy \
       --reward-path ../safe_rl_data/rewards_base_e_exponential_len10.npy \
       --reward_type e_exp \
       --shuffle

python offline_dqn.py \
       --exp-name offline_dqn_e_exp_no_shuffle_500000 \
       --track \
       --wandb-project-name soft_limb \
       --total-timesteps 500000 \
       --target-network-frequency 500 \
       --train-frequency 10 \
       --batch-size 512 \
       --network-type LSTM \
       --state-path ../safe_rl_data/states_len10.npy \
       --next-state-path ../safe_rl_data/next_states_len10.npy \
       --action-path ../safe_rl_data/actions_len10.npy \
       --reward-path ../safe_rl_data/rewards_base_e_exponential_len10.npy \
       --reward_type e_exp

# 1 + gamma aka (lam)
python offline_dqn.py \
       --exp-name offline_dqn_exp_no_act_no_shuffle_500000 \
       --track \
       --wandb-project-name soft_limb \
       --total-timesteps 500000 \
       --target-network-frequency 500 \
       --train-frequency 10 \
       --batch-size 512 \
       --network-type LSTM \
       --state-path ../safe_rl_data/states_len10_no_action.npy \
       --next-state-path ../safe_rl_data/next_states_no_action_len10.npy \
       --action-path ../safe_rl_data/actions_len10.npy \
       --reward-path ../safe_rl_data/rewards_exponential_len10.npy \
       --reward_type exp

python offline_dqn.py \
       --exp-name offline_dqn_exp_no_act_shuffle_500000 \
       --track \
       --wandb-project-name soft_limb \
       --total-timesteps 500000 \
       --target-network-frequency 500 \
       --train-frequency 10 \
       --batch-size 512 \
       --network-type LSTM \
       --state-path ../safe_rl_data/states_len10_no_action.npy \
       --next-state-path ../safe_rl_data/next_states_no_action_len10.npy \
       --action-path ../safe_rl_data/actions_len10.npy \
       --reward-path ../safe_rl_data/rewards_exponential_len10.npy \
       --reward_type exp \
       --shuffle

python offline_dqn.py \
       --exp-name offline_dqn_exp_shuffle_500000 \
       --track \
       --wandb-project-name soft_limb \
       --total-timesteps 500000 \
       --target-network-frequency 500 \
       --train-frequency 10 \
       --batch-size 512 \
       --network-type LSTM \
       --state-path ../safe_rl_data/states_len10.npy \
       --next-state-path ../safe_rl_data/next_states_len10.npy \
       --action-path ../safe_rl_data/actions_len10.npy \
       --reward-path ../safe_rl_data/rewards_exponential_len10.npy \
       --reward_type exp \
       --shuffle

python offline_dqn.py \
       --exp-name offline_dqn_exp_no_shuffle_500000 \
       --track \
       --wandb-project-name soft_limb \
       --total-timesteps 500000 \
       --target-network-frequency 500 \
       --train-frequency 10 \
       --batch-size 512 \
       --network-type LSTM \
       --state-path ../safe_rl_data/states_len10.npy \
       --next-state-path ../safe_rl_data/next_states_len10.npy \
       --action-path ../safe_rl_data/actions_len10.npy \
       --reward-path ../safe_rl_data/rewards_exponential_len10.npy \
       --reward_type exp