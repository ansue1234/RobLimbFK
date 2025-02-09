import torch 
import torch.nn as nn
import torch.nn.functional as F
import argparse
import yaml
import gymnasium as gym
import numpy as np
from robo_limb_rl.envs.limb_env import LimbEnv
from tqdm import tqdm
from robo_limb_rl.scripts.sac_cont_action import Actor, make_env

def load_env(seed, render_mode='human'):
    env = gym.make('LimbEnv-v0', config_path='./yaml/testing_limb_env.yml', render_mode=render_mode, seed=seed)
    return env
    
# Only Safe Env for now, Nominal policy is a random policy, supports only MLP for now
if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = 1
    # Prep Environment
    env = load_env(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Policy
    # envs = gym.vector.SyncVectorEnv([lambda: load_env(seed, render_mode=None)])
    # actor = Actor(envs).to(device)
    # actor_params, _, _ = torch.load('../policies/LimbEnv-v0__sac_1M_new__1__1728083577/sac_1M_new.cleanrl_model', map_location=device, weights_only=True)
    # actor.load_state_dict(actor_params)
    # actor.eval()

    # Rollout
    # x = 40*np.sin(np.linspace(0, 2*np.pi, 360))
    # y = 40*np.cos(np.linspace(0, 2*np.pi, 360))
    # path_to_track = np.vstack((x, y)).T
    
    obs, _ = env.reset()
    # env.set_goal(path_to_track[0])
    for i in range(800):
        # obs = torch.tensor(obs).to(torch.float32).to(device).unsqueeze(0)
        # action, _, _ = actor.get_action(obs)
        # obs, _, done, _, _ = env.step(action.detach().cpu().numpy()[0])
        obs, _, done, _, _ = env.step(env.action_space.sample())
        if done:
            obs, _ = env.reset()
        # print("Observation: ", obs)
        # print("Goal: ", path_to_track[i])
        # if done:
        # obs, _ = env.reset()
        # print(env.goal)
            # env.set_goal(path_to_track[i])
        # if np.linalg.norm(obs[:2] - path_to_track[i]) < 3:
        #     i += 1
        #     print(f"Reached point {i}")
        #     env.set_goal(path_to_track[i])
        # elif done:
        #     obs, _ = env.reset()
        #     env.set_goal(path_to_track[i])