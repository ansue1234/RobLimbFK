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
                 device):
        
        self.device = device
        self.batch_size = batch_size
        self.state = np.load(state_path)
        self.next_state = np.load(next_state_path)
        self.actions = np.load(action_path)
        self.rewards = np.load(reward_path)
        
        self.num_samples = self.state.shape[0]
        self.seq_len = self.state.shape[1]
        self.obs_space = self.state.shape[2]
        self.action_space = num_throttle_actions**2 # 21 x throttle, 21 y throttle
        self.num_throttle_actions = num_throttle_actions
        self.current_batch = 0
        self.n_batches = self.num_samples // self.batch_size
        
    def get_data(self, shuffle=True):
        if not shuffle:
            if self.current_batch == self.n_batches:
                self.current_batch = 0
            
            if (self.current_batch + 1)*self.batch_size > self.num_samples:
                rewards = self.rewards[self.current_batch*self.batch_size:]
                dones = rewards <= 0
                return torch.tensor(self.state[self.current_batch*self.batch_size:]).to(self.device),\
                        torch.tensor(self.next_state[self.current_batch*self.batch_size:]).to(self.device),\
                        torch.tensor(self.actions[self.current_batch*self.batch_size:]).to(self.device),\
                        torch.tensor(rewards).to(self.device), \
                        torch.tensor(dones).to(self.device)
            else:
                rewards = self.rewards[self.current_batch*self.batch_size:(self.current_batch + 1)*self.batch_size]
                dones = rewards <= 0
                return torch.tensor(self.state[self.current_batch*self.batch_size:(self.current_batch + 1)*self.batch_size]).to(self.device),\
                        torch.tensor(self.next_state[self.current_batch*self.batch_size:(self.current_batch + 1)*self.batch_size]).to(self.device),\
                        torch.tensor(self.actions[self.current_batch*self.batch_size:(self.current_batch + 1)*self.batch_size]).to(self.device),\
                        torch.tensor(rewards).to(self.device), \
                        torch.tensor(dones).to(self.device)
        else:
            index = np.randomInt(0, len(self.state) - self.batch_size)
            rewards = self.rewards[index:index+self.batch_size]
            dones = rewards <= 0
            return torch.tensor(self.state[index:index+self.batch_size]).to(self.device),\
                    torch.tensor(self.next_state[index:index+self.batch_size]).to(self.device),\
                    torch.tensor(self.actions[index:index+self.batch_size]).to(self.device),\
                    torch.tensor(rewards).to(self.device), \
                    torch.tensor(dones).to(self.device)
    
    def get_dims(self):
        return self.obs_space, self.action_space
    
    def get_action(self, actions):
        shifted_actions = actions + self.num_throttle_actions//2
        indices = shifted_actions[:, 0] * 21 + shifted_actions[:, 1]
        return indices