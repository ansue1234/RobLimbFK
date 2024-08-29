import gymnasium as gym
from robo_limb_rl.envs.limb_env import LimbEnv, SafeLimbEnv
from tqdm import tqdm

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
    env = gym.make('SafeLimbEnv-v0', config_path='../scripts/yaml/safe_limb_env.yml', render_mode=None)

    obs, _ = env.reset()
    for i in tqdm(range(1000)):
        action = env.action_space.sample()  # Sample a random action
        obs, reward, done, _, _ = env.step(action)
        if done:
            print(i, reward)
            break
    env.close()