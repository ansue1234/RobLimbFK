import gymnasium as gym

class RandomPolicy:
    def __init__(self, env, seed=1):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.seed = seed
    
    def model(self, obs):
        return self.action_space.sample()