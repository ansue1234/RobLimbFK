import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

class LimbEnv(gym.Env):
    def __init__(self, config):
        super(LimbEnv, self).__init__()
        self.model_path = config['model_path']
        self.model = load_model(self.model_path)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
