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


class TrajData():
    def __init__(self, state, next_state, action, reward, done):
        self.observations = state
        self.next_observations = next_state
        self.actions = action[-1]
        self.rewards = reward[-1]
        self.dones = done[-1]

class TrajReplayBuffer():
    def __init__(self, max_size, state_dim, action_dim, seq_len, device, sample_type='random_bootstrap'):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.seq_len = seq_len
        self.size = 0
        self.trajs = []
        self.trajs_next_states = []
        self.trajs_actions = []
        self.trajs_rewards = []
        self.trajs_dones = []
        
        self.current_traj = []
        self.current_traj_next_states = []
        self.current_traj_actions = []
        self.current_traj_rewards = []
        self.current_traj_dones = []
        self.sample_type = sample_type
    
    def add(self, state, next_state, action, reward, done):

        self.current_traj.append(state)
        self.current_traj_next_states.append(next_state)
        self.current_traj_actions.append(action)
        self.current_traj_rewards.append(reward)
        self.current_traj_dones.append(done)
        
        if done:
            self.trajs.append(self.current_traj)
            self.trajs_next_states.append(self.current_traj_next_states)
            self.trajs_actions.append(self.current_traj_actions)
            self.trajs_rewards.append(self.current_traj_rewards)
            self.trajs_dones.append(self.current_traj_dones)
            
            self.current_traj = []
            self.current_traj_next_states = []
            self.current_traj_actions = []
            self.current_traj_rewards = []
            self.current_traj_dones = []
            
            if self.size < self.max_size:
                self.size += 1
            else:
                self.size = self.max_size
                self.trajs.pop(0)
                self.trajs_next_states.pop(0)
                self.trajs_actions.pop(0)
                self.trajs_rewards.pop(0)
                self.trajs_dones.pop(0)
                
    
    def sample(self):
        traj_index = np.random.randint(0, self.size)
        traj_states = np.array(self.trajs[traj_index])
        traj_next_states = np.array(self.trajs_next_states[traj_index])
        actions = np.array(self.trajs_actions[traj_index])
        rewards = np.array(self.trajs_rewards[traj_index])
        dones = np.array(self.trajs_dones[traj_index])
        
        if self.sample_type == 'random_bootstrap' and len(traj_states) > self.seq_len:
            traj_start_point = np.random.randint(0, len(traj_states) - self.seq_len)
            # print(traj_states.shape)
            # print(traj_next_states.shape)
            # print(actions.shape)
            # print(rewards.shape)
            # print(dones.shape)
            return TrajData(torch.tensor(traj_states[traj_start_point:traj_start_point+self.seq_len]).to(self.device),
                            torch.tensor(traj_next_states[traj_start_point:traj_start_point+self.seq_len]).to(self.device),
                            torch.tensor(actions[traj_start_point:traj_start_point+self.seq_len]).to(self.device),
                            torch.tensor(rewards[traj_start_point:traj_start_point+self.seq_len]).to(self.device),
                            torch.tensor(dones[traj_start_point:traj_start_point+self.seq_len]).to(self.device))
        else:
            return TrajData(torch.tensor(traj_states).to(self.device),
                            torch.tensor(traj_next_states).to(self.device),
                            torch.tensor(actions).to(self.device),
                            torch.tensor(rewards).to(self.device),
                            torch.tensor(dones).to(self.device))
        # traj_start_point = np.random.randint(0, len(traj_states) - self.seq_len)