import gymnasium as gym
from robo_limb_rl.envs.limb_env import LimbEnv, SafeLimbEnv
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    
    # env = gym.make('LimbEnv-v0', config_path='../scripts/yaml/default_limb_env.yml', render_mode=None)

    # obs = env.reset()
    # for i in tqdm(range(1000)):
    #     action = env.action_space.sample()  # Sample a random action
    #     obs, reward, done, _, _ = env.step(action)
    #     if done:
    #         print(i)
    #         break
    # env.close()
    env = gym.make('SafeLimbEnv-v0', config_path='../scripts/yaml/safe_limb_env_discrete.yml', render_mode='human')

    obs, _ = env.reset()
    env.set_state(np.zeros(4).astype(np.float32))
    for i in tqdm(range(1000)):
        # action = env.action_space.sample()  # Sample a random action
        
        obs, reward, done, _, _ = env.step(0)
        if done:
            print(i, reward)
            break
    env.close()