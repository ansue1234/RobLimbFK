import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from robo_limb_ml.models.fk_lstm import FK_LSTM
from robo_limb_ml.models.fk_seq2seq import FK_SEQ2SEQ

class LimbEnv(gym.Env):
    def __init__(self, config):
        super(LimbEnv, self).__init__()
        self.model_path = config.get('model_path', "")
        self.model_type = config.get('model_type', "SEQ2SEQ")
        self.viz_type = config.get('viz_type', 'scatter')
        self.input_dim = config.get('input_dim', 6)
        self.output_dim = config.get('output_dim', 4)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.seq_len = config.get('seq_len', 100)
        self.num_layers = config.get('num_layers', 3)
        self.attention = config.get('attention', False)
        self.action_type = config.get('action_type', 'continuous')
        self.stateful = config.get('stateful', False)
        self.vel_type = config.get('vel_type', 'computed')
        self.dt = config.get('dt', 0.075) # 75 ms
        self.theta_limit = config.get('theta_limit', 100)
        self.goal_tolerance = config.get('goal_tolerance', 1)
        
        # Setting up model
        self.hidden = (torch.zeros(self.num_layers, 1, self.hidden_dim),
                       torch.zeros(self.num_layers, 1, self.hidden_dim))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(self.model_path)
        
        if self.action_type == 'continuous':
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        else:
            self.action_space = gym.spaces.Discrete(441)
        self.observation_space = gym.spaces.Box(low=-self.theta_limit, high=self.theta_limit, shape=(4,))
        self.reset()
        
        # setting up visualizations
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_xlabel('X Angle (degrees)')
        self.ax.set_ylabel('Y Angle (degrees)')
        self.ax.set_title('Live State Visualization')
        
        # Initialize scatter plot
        if self.viz_type == 'scatter':
            self.scatter = self.ax.scatter([], [], c='blue', s=100)  # 's' is the size of the scatter points
        elif self.viz_type == 'line':
            self.line, = self.ax.plot([], [], 'r-', markersize=10)
        
        self.goal_plot = self.ax.scatter(self.goal[0], self.goal[1], c='red', s=100, marker='x')  # Goal marker

    def set_state(self, state):
        self.state = state
    
    def set_goal(self, goal):
        self.goal = goal
    
    def get_state(self):
        return self.state
    
    def get_goal(self):
        return self.goal
    
    def get_data(self):
        return self.data
    
    def reset(self):
        self.state = np.random.uniform(-60, 60, 4)
        self.goal = np.random.uniform(-60, 60, 2)
        self.data = torch.tensor(np.concatenate(self.state, np.array([0.0, 0.0])))
        return self.state
    
    def step(self, action):
        # prep data
        current_data_entry = torch.tensor(np.concatenate((self.state, action)))
        self.data = torch.cat((self.data, current_data_entry))
        if self.data.shape[0] > self.seq_len:
            self.data = self.data[1:]
        
        # model inference
        hn, cn = self.hidden
        prev_state = self.state.copy()
        if self.stateful:
            if self.model_type == 'LSTM':
                delta_states, hn, cn = self.model(self.data.unsqueeze(0), hn, cn)
            elif self.model_type == 'SEQ2SEQ':
                delta_states, self.hidden = self.model(self.data.unsqueeze(0), None, self.hidden, mode='test')
        else:
            if self.model_type == 'LSTM':
                delta_states, _, _ = self.model(self.data.unsqueeze(0), hn, cn)
            else:
                delta_states, _ = self.model(self.data.unsqueeze(0), None, self.hidden, mode='test')
        delta_states = delta_states.squeeze().detach().cpu().numpy()
        
        # state updates
        self.state = self.state + delta_states
        if self.vel_type == 'computed':
            vel = (self.state[:2] - prev_state[:2]) / self.dt
            self.state[2:] = vel
        # compute reward
        reward = self.compute_reward(self.state, self.goal, self.goal_tolerance)
        
        #termination condition
        done = self.check_termination()
    
        return self.state, reward, done, False, {} 
    
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
        
    def load_model(self, model_path):
        if self.model_type == 'LSTM':
            self.model = FK_LSTM(input_size=self.input_dim,
                                 hidden_size=self.hidden_dim,
                                 num_layers=self.num_layers,
                                 batch_size=1,
                                 output_size=4,
                                 device=self.device,
                                 batch_first=True).to(device=self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.h0, self.model.c0 = self.hidden
        elif self.model_type == 'SEQ2SEQ':
            self.model = FK_SEQ2SEQ(input_size=self.input_dim,
                                    embedding_size=self.hidden_dim,
                                    num_layers=self.num_layers,
                                    batch_size=1,
                                    output_size=4,
                                    device=self.device,
                                    batch_first=True,
                                    encoder_type='LSTM',
                                    decoder_type='LSTM',
                                    attention=self.attention,
                                    pred_len=1,
                                    teacher_forcing_ratio=0.0).to(device=self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.encoder.h0, self.model.encoder.c0 = self.hidden
        self.model.eval()
    
    def compute_reward(self, state, goal, action):
        # idea from openai gym's reacher-v2 reward function
        return - np.linalg.norm(state[:2] - goal) - np.sum(action**2)
    
    def check_termination(self):
        if self.state[0] > self.theta_limit or self.state[0] < -self.theta_limit or self.state[1] > self.theta_limit or self.state[1] < -self.theta_limit:
            return True
        if self.state[2] > self.theta_limit or self.state[2] < -self.theta_limit or self.state[3] > self.theta_limit or self.state[3] < -self.theta_limit:
            return True
        if np.linalg.norm(self.state[:2] - self.goal) < 1:
            return True
        return False
    
    def sample_states(self, num_samples):
        return np.random.uniform(-self.theta_limit, self.theta_limit, (num_samples, 4))
    
    def close(self):
        plt.close(self.fig)

class SafeLimbEnv(LimbEnv):
    def __init__(self, config):
        super(SafeLimbEnv, self).__init__(config)
        self.safe_zone = config.get('safe_zone', 40)
        self.gamma = config.get('gamma', 0.99)
        self.alpha = config.get('alpha', 0.01)
        self.reward_type = config.get('reward_type', 'reg')
        self.t = 0
        self.hit_unsafe = False
    
    def check_termination(self):
        if self.state[2] > self.theta_limit or self.state[2] < -self.theta_limit or self.state[3] > self.theta_limit or self.state[3] < -self.theta_limit:
            return True
        if np.linalg.norm(self.state[:2]) > self.safe_zone:
            return True
        return False
    
    def compute_reward(self, state, goal, action):
        l = self.distance(state)
        if self.reward_type == 'reg':
            if np.linalg.norm(state[:2]) <= self.safe_zone:
                return l
            else:
                if not self.hit_unsafe:
                    self.hit_unsafe = True
                    return -1/((1-self.gamma)*self.gamma**self.t)
                else:
                    return -1
        elif self.reward_type == 'exp':
            if np.linalg.norm(state[:2]) <= self.safe_zone:
                return np.exp(-np.power(self.gamma, self.t)/l)
            else:
                return 0
        elif self.reward_type == 'base_exp':
            if np.linalg.norm(state[:2]) <= self.safe_zone:
                return np.power(1 + self.alpha, -np.power(self.alpha, self.t)/l)
            else:
                return 0
            
    def distance(self, states):
        if states.ndim == 1:
            return (self.safe_zone - np.linalg.norm(states[:2]))/self.safe_zone
        else:
            return (self.safe_zone - np.linalg.norm(states[:, :2], axis=1))/self.safe_zone
        
    def step(self, action):
        self.t += 1
        return super(SafeLimbEnv, self).step(action)
    
    def reset(self):
        self.t = 0
        self.hit_unsafe = False
        super(SafeLimbEnv, self).reset()
        self.state = np.random.uniform(-self.safe_zone, self.safe_zone, 4)
    
    def is_safe(self, states):
        if states.ndim == 1:
            if np.linalg.norm(states[:2]) <= self.safe_zone:
                return True
        else:
            return np.linalg.norm(states[:, :2], axis=1) <= self.safe_zone
    