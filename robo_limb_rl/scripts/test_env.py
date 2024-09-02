import gymnasium as gym
from robo_limb_rl.envs.limb_env import LimbEnv, SafeLimbEnv
from tqdm import tqdm
import numpy as np

def make_env(env_id, seed, config_path):
    def thunk():
        env = gym.make(env_id, seed=seed, config_path=config_path, render_mode=None)
        print("env id", env_id)
        print("config path", config_path)
        print("env max steps", env.spec.max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

if __name__ == "__main__":
    
    nom_env = gym.make('LimbEnv-v0', config_path='../scripts/yaml/default_limb_env_discrete.yml', render_mode=None, seed=1)

    # obs = env.reset()
    # for i in tqdm(range(1000)):
    #     action = env.action_space.sample()  # Sample a random action
    #     obs, reward, done, _, _ = env.step(action)
    #     if done:
    #         print(i)
    #         break
    # env.close()
    safe_env = gym.make('SafeLimbEnv-v0', config_path='../scripts/yaml/safe_limb_env_discrete.yml', render_mode=None, seed=1)
    envs = gym.vector.SyncVectorEnv(
        [make_env('SafeLimbEnv-v0', i, config_path='../scripts/yaml/safe_limb_env_simple_discrete.yml') for i in range(1)],
    )
    print("safe_env", safe_env.spec.max_episode_steps)
    _, _ = nom_env.reset()
    _, _ = safe_env.reset()
    nom_env.set_state(np.zeros(4).astype(np.float32))
    # nom_env.set_goal(np.zeros(2).astype(np.float32)+5)
    safe_env.set_state(np.zeros(4).astype(np.float32))
    for i in tqdm(range(1000)):
        # action = env.action_space.sample()  # Sample a random action
        
        safe_obs, safe_reward, safe_done, _, _ = safe_env.step(221)
        nom_obs, nom_reward, nom_done, _, _ = nom_env.step(221)
        print("safe_obs", safe_obs)
        print("nom_obs", nom_obs)
        if safe_done:
            print(i, safe_reward, nom_reward)
            break
        if i % 10 == 0:
            print("resetting")
            nom_env.reset(seed=4)
            safe_env.reset(seed=4)
            nom_env.set_state(np.zeros(4).astype(np.float32))
            # nom_env.set_goal(np.zeros(2).astype(np.float32)+5)
            safe_env.set_state(np.zeros(4).astype(np.float32))
        # if nom_done:
        #     print(i, nom_reward)
        #     break
    safe_env.close()
    nom_env.close()