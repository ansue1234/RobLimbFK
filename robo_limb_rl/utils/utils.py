import torch 
import numpy as np

class DataLoader():
    def __init__(self,
                 state_path,
                 next_state_path,
                 action_path,
                 reward_path,
                 num_throttle_actions,
                 batch_size,
                 device,
                 shuffle=True):
        
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.state = np.load(state_path).astype(np.float32)
        self.next_state = np.load(next_state_path).astype(np.float32)
        self.actions = np.load(action_path).astype(np.float32)
        self.rewards = np.load(reward_path).astype(np.float32)
        
        self.num_samples = self.state.shape[0]
        self.seq_len = self.state.shape[1]
        self.obs_space = self.state.shape[2]
        self.action_space = num_throttle_actions**2 # 21 x throttle, 21 y throttle
        self.num_throttle_actions = num_throttle_actions
        self.current_batch = 0
        self.n_batches = self.num_samples // self.batch_size
        
        if not self.shuffle:
            self.num_samples = (self.state.shape[0]//self.batch_size)*self.batch_size
            self.state = self.state[:self.num_samples]
            self.next_state = self.next_state[:self.num_samples]
            self.actions = self.actions[:self.num_samples]
            self.rewards = self.rewards[:self.num_samples]
            
    def get_data(self):
        if not self.shuffle:
            if self.current_batch == self.n_batches:
                self.current_batch = 0
            
            if (self.current_batch + 1)*self.batch_size > self.num_samples:
                rewards = self.rewards[self.current_batch*self.batch_size:]
                dones = rewards <= 0
                return torch.tensor(self.state[self.current_batch*self.batch_size:]).to(self.device),\
                        torch.tensor(self.next_state[self.current_batch*self.batch_size:]).to(self.device),\
                        torch.tensor(self.actions[self.current_batch*self.batch_size:]).to(self.device),\
                        torch.tensor(rewards).to(self.device), \
                        torch.tensor(dones).to(self.device), self.current_batch + 1 == self.n_batches
            else:
                rewards = self.rewards[self.current_batch*self.batch_size:(self.current_batch + 1)*self.batch_size]
                dones = rewards <= 0
                return torch.tensor(self.state[self.current_batch*self.batch_size:(self.current_batch + 1)*self.batch_size]).to(self.device),\
                        torch.tensor(self.next_state[self.current_batch*self.batch_size:(self.current_batch + 1)*self.batch_size]).to(self.device),\
                        torch.tensor(self.actions[self.current_batch*self.batch_size:(self.current_batch + 1)*self.batch_size]).to(self.device),\
                        torch.tensor(rewards).to(self.device), \
                        torch.tensor(dones).to(self.device), self.current_batch + 1 == self.n_batches
        else:
            index = np.random.randint(0, len(self.state) - self.batch_size)
            rewards = self.rewards[index:index+self.batch_size]
            dones = rewards <= 0
            return torch.tensor(self.state[index:index+self.batch_size], ).to(self.device),\
                    torch.tensor(self.next_state[index:index+self.batch_size]).to(self.device),\
                    torch.tensor(self.actions[index:index+self.batch_size]).to(self.device),\
                    torch.tensor(rewards).to(self.device), \
                    torch.tensor(dones).to(self.device), self.current_batch + 1 == self.n_batches
    
    def get_dims(self):
        return self.obs_space, self.action_space
    
    def get_action(self, actions):
        shifted_actions = actions + self.num_throttle_actions//2
        indices = shifted_actions[:, 0] * 21 + shifted_actions[:, 1]
        return indices


class TrajReplayBuffer():
    def __init__(self, max_size, state_dim, action_dim, seq_len, device):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.seq_len = seq_len
        self.size = 0
        self.trajs = []
        self.trajs_next_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.current_traj = []
        self.current_traj_next_states = []
    
    def add(self, state, next_state, action, reward, done):
        self.current_traj.append(state)
        self.current_traj_next_states.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        if done:
            self.trajs.append(self.current_traj)
            self.trajs_next_states.append(self.current_traj_next_states)
            self.current_traj = []
            self.current_traj_next_states = []
        if self.size < self.max_size:
            self.size += 1
        else:
            self.trajs.pop(0)
            self.trajs_next_states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
    
    def sample(self):
        