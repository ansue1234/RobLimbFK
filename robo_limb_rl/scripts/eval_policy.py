import torch 
import argparse
import yaml
import gymnasium as gym


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path config file')
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Prep Environment
    safe_env_config = config.get('safe_env', None)
    if safe_env_config:
        safe_env_config_path = safe_env_config['config_path']
        safe_env = gym.make(safe_env_config['env_name'], config_path=safe_env_config_path, render_mode=safe_env_config['render_mode'])
    
    nom_env_config = config.get('nom_env', None)
    if nom_env_config:
        nom_env_config_path = nom_env_config['config_path']
        nom_env = gym.make(nom_env_config['env_name'], config_path=nom_env_config_path, render_mode=nom_env_config['render_mode'])
    
    # Load Policy
    policy_config = config.get('policy', None)
    if policy_config:
        policy_types = policy_config['policy_types']
        policy_path = policy_config['policy_paths']
        algo_type = policy_config['algo_type']
        policy_dict = {}
        
        for i, (policy_type, policy_path) in enumerate(zip(policy_types, policy_path)):
            if policy_type == 'QNet_MLP':
                policy = torch.load(policy_path)
            elif policy_type == 'QNet_LSTM':
                policy = torch.load(policy_path)
            else:
                raise ValueError(f'Policy type {policy_type} not supported')
            
            if algo_type == 'DQN' and i == 0:
                policy_dict['qf'] = policy
            elif algo_type == 'SAC':
                if i == 0:
                    policy_dict['qf1'] = policy
                if i == 1:
                    policy_dict['qf2'] = policy
                else:
                    policy_dict['actor'] = policy
    
        