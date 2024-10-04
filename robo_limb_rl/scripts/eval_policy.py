import torch 
import argparse
import yaml
import gymnasium as gym
import numpy as np
from robo_limb_rl.arch.Q_net import QNet_MLP, QNet_LSTM
from robo_limb_rl.utils.policies import RandomPolicy
from robo_limb_rl.envs.limb_env import LimbEnv, SafeLimbEnv
from tqdm import tqdm

def get_action(policy_dict, obs, device):
    with torch.no_grad():
        obs = torch.tensor(obs).to(device)
        if policy_dict['algo'] == 'DQN':
            q_values = policy_dict['qf'](obs)
            return torch.argmax(q_values).item(), torch.max(q_values).item()
        elif policy_dict['algo'] == 'SAC':
            q_values = policy_dict['qf1'](obs)
            return torch.argmax(q_values).item(), torch.max(q_values).item()
        else:
            return policy_dict['model'].model(obs), None

def load_policy(policy_config, env, device, safe_env=None):
    if policy_config:
        policy_types = policy_config['policy_types']
        policy_path = policy_config['policy_paths']
        algo_type = policy_config['algo_type']
        policy_dict = {}
        
        for i, (policy_type, policy_path) in enumerate(zip(policy_types, policy_path)):
            if policy_type == 'QNet_MLP':
                # print("obs shape", env.action_space.shape)
                policy = QNet_MLP(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n, reward_type=policy_config['reward_type']).to(device)
                policy.load_state_dict(torch.load(policy_path, map_location=device))
                policy.eval()
            elif policy_type == 'QNet_LSTM':
                policy = torch.load(policy_path)
            else:
                raise ValueError(f'Policy type {policy_type} not supported')
            
            if algo_type == 'DQN' and i == 0:
                policy_dict['algo'] = 'DQN'
                policy_dict['qf'] = policy
            elif algo_type == 'SAC':
                if i == 0:
                    policy_dict['algo'] = 'SAC'
                    policy_dict['qf1'] = policy
                if i == 1:
                    policy_dict['qf2'] = policy
                else:
                    policy_dict['actor'] = policy
        return policy_dict
    return {'algo': 'Random',
            'model': RandomPolicy(safe_env)}

def load_env(env_config, seed):
    if env_config:
        env = gym.make(env_config['env_name'], config_path=env_config['config_path'], render_mode=env_config['render_mode'], seed=seed)
        return env
    return None

def rollout(safe_env, nom_env, safe_policy_dict, nom_policy_dict, device, max_steps=1000, seed=1, intervention=True):
    nom_obs, _ = nom_env.reset(seed=seed)
    safe_obs, _ = safe_env.reset(seed=seed)
    nom_env.set_state(np.zeros(4).astype(np.float32))
    nom_env.set_goal(np.zeros(2).astype(np.float32)+110)
    safe_env.set_state(np.zeros(4).astype(np.float32))
    
    nom_rewards = 0
    ep_len = 0
    ep_ended = False
    for i in tqdm(range(max_steps)):
        # action = safe_env.action_space.sample()  # Sample a random action
        safe_action, safe_q_val = get_action(safe_policy_dict, safe_obs, device)
        nom_action, _ = get_action(nom_policy_dict, nom_obs, device)
        if safe_q_val > 0 and intervention:
            action = safe_action
        else:
            action = nom_action
        action = torch.tensor(action).to(torch.float32)
        safe_obs, _, safe_done, _, _ = safe_env.step(action)
        nom_obs, nom_reward, _, _, _ = nom_env.step(action)
        nom_rewards += nom_reward
        if safe_done:
            if not ep_ended:
                ep_len = i + 1
                ep_ended = True
                break
            if intervention:
                break
    if not ep_ended:
        ep_len = i + 1
    safe_env.close()
    nom_env.close()
    return ep_len, nom_rewards
    
    
    
# Only Safe Env for now, Nominal policy is a random policy, supports only MLP for now
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path config file')
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = config.get('seed', 1)
    
    # Prep Environment
    safe_env_config = config.get('safe_env', None)
    safe_env = load_env(safe_env_config, seed)
    nom_env_config = config.get('nom_env', None)
    nom_env = load_env(nom_env_config, seed)
    
    # Load Policy
    safe_policy_config = config.get('safety_policy', None)
    safe_policy_dict = load_policy(safe_policy_config, safe_env, device)
    nom_env_config = config.get('nominal_policy', None)
    nom_policy_dict = load_policy(nom_env_config, nom_env, device, safe_env=safe_env)
    
    if safe_env is None or safe_policy_dict is None:
        raise ValueError('Safe Env or Safe Policy not loaded properly')
    if nom_env is None or nom_policy_dict is None:
        print('Nominal Env or Nominal Policy not loaded properly')
        print('Using Random Policy')
        nom_env = load_env(safe_env_config, seed)
    num_passed_episodes = 0
    total_rewards = 0
    max_steps = config.get('max_steps', 200)
    num_rollouts = config.get('num_rollouts', 100)
    for i in tqdm(range(num_rollouts)):
        steps, rewards = rollout(safe_env, nom_env, safe_policy_dict, nom_policy_dict, device, max_steps=max_steps, seed=i)
        if steps >= max_steps:
            num_passed_episodes += 1
        total_rewards += rewards
        print(f'Rollout {i}: Steps: {steps}, Rewards: {rewards}')
    print('With Intervention:')
    print(f'Success Rate:{num_passed_episodes}/{num_rollouts}, Avg Rewards: {total_rewards/num_rollouts}')
    
    # num_passed_episodes = 0
    # total_rewards = 0
    # for i in tqdm(range(num_rollouts)):
    #     steps, rewards = rollout(safe_env, nom_env, safe_policy_dict, nom_policy_dict, device, max_steps=max_steps, seed=i, intervention=False)
    #     if steps >= max_steps:
    #         num_passed_episodes += 1
    #     total_rewards += rewards
    # print('Without Intervention:')
    # print(f'Success Rate:{num_passed_episodes}/{num_rollouts}, Avg Rewards: {total_rewards/num_rollouts}')
        