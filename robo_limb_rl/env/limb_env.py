import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

class LimbEnv(gym.Env):
    def __init__(self, config):
        super(LimbEnv, self).__init__()
        self.model_path = config['model_path']
        self.model_type = config['model_type']
        self.viz_type = config['viz_type']
        self.model = self._load_model(self.model_path)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        self.reset()
        
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_xlabel('X Angle (degrees)')
        self.ax.set_ylabel('Y Angle (degrees)')
        self.ax.set_title('Live State Visualization')
        
        self.state = np.zeros(4)
        self.goal = np.zeros(2)
        
        # Initialize scatter plot
        if self.viz_type == 'scatter':
            self.scatter = self.ax.scatter([], [], c='blue', s=100)  # 's' is the size of the scatter points
        elif self.viz_type == 'line':
            self.line, = self.ax.plot([], [], 'r-', markersize=10)
        
        self.goal_plot = self.ax.scatter(self.goal[0], self.goal[1], c='red', s=100, marker='x')  # Goal marker


    def reset(self):
        self.state = np.random.rand(4)
        self.goal = np.random.rand(2)
        return self.state
    
    def set_state(self, state):
        self.state = state
    
    def set_goal(self, goal):
        self.goal = goal
    
    def step(self, action):
        self.state = self.model.predict(self.state, action)
        return self.state, 0, False, {} 
    
    def render(self, mode='human'):
        if mode == 'human':
            # Update the scatter plot with the current state
            if self.viz_type == 'scatter':
                self.scatter.set_offsets([self.state[0], self.state[1]])
            elif self.viz_type == 'line':
                self.line.set_xdata([self.state[0]])
                self.line.set_ydata([self.state[1]])
            self.goal_plot.set_offsets([self.goal[0], self.goal[1]])
            self.ax.relim()
            self.ax.autoscale_view()
            plt.draw()
            plt.pause(0.1)
        
    def _load_model(self, model_path):
        
        return torch.load(model_path)