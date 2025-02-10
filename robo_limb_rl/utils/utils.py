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
            return torch.tensor(self.state[index:index+self.batch_size]).to(self.device),\
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
        self.actions = action
        self.rewards = reward
        self.dones = done

class TrajReplayBuffer():
    def __init__(self, max_size, device):
        self.max_size = max_size
        self.device = device
        self.size = 0
        self.trajs = []
        self.trajs_next_states = []
        self.trajs_actions = []
        self.trajs_rewards = []
        self.trajs_dones = []
        self.trajs_lengths = [0]
        self.trajs_info = []
        
        self.current_traj = []
        self.current_traj_next_states = []
        self.current_traj_actions = []
        self.current_traj_rewards = []
        self.current_traj_dones = []
        self.current_traj_info = []
    
    def add(self, state, next_state, action, reward, done, truncated, infos):
        state = state[0]
        next_state = next_state[0]
        action = action[0]
        reward = reward[0]
        done = done[0]
        truncated = truncated[0]
        
        self.current_traj.append(state)
        self.current_traj_next_states.append(next_state)
        self.current_traj_actions.append(action)
        self.current_traj_rewards.append(reward)
        self.current_traj_dones.append(done)
        self.trajs_lengths[-1] += 1
        
        if done or truncated:
            self.trajs.append(torch.Tensor(self.current_traj))
            self.trajs_next_states.append(torch.Tensor(self.current_traj_next_states))
            self.trajs_actions.append(torch.Tensor(self.current_traj_actions))
            self.trajs_rewards.append(torch.Tensor(self.current_traj_rewards))
            self.trajs_dones.append(torch.Tensor(self.current_traj_dones))
            
            
            self.current_traj = []
            self.current_traj_next_states = []
            self.current_traj_actions = []
            self.current_traj_rewards = []
            self.current_traj_dones = []
            self.trajs_lengths.append(0)
            
            if self.size < self.max_size:
                self.size += 1
            else:
                self.size = self.max_size
                self.trajs.pop(0)
                self.trajs_next_states.pop(0)
                self.trajs_actions.pop(0)
                self.trajs_rewards.pop(0)
                self.trajs_dones.pop(0)
                self.trajs_lengths.pop(0)
                
    
    def sample(self, batch_size=1):
        if self.size < 1:
            sampled_trajs = torch.tensor(np.array(batch_size * [self.current_traj])).type(torch.float).to(self.device)
            sampled_trajs_next_states = torch.tensor(np.array(batch_size * [self.current_traj_next_states])).type(torch.float).to(self.device)
            sampled_trajs_actions = torch.tensor(batch_size * [self.current_traj_actions[-1]]).type(torch.float).to(self.device)
            sampled_trajs_rewards = torch.tensor(batch_size * [self.current_traj_rewards[-1:]]).type(torch.float).to(self.device)
            sampled_trajs_dones = torch.tensor(batch_size * [self.current_traj_dones[-1:]]).type(torch.float).to(self.device)
            return TrajData(sampled_trajs, sampled_trajs_next_states, sampled_trajs_actions, sampled_trajs_rewards, sampled_trajs_dones)
        
        # if self.size < batch_size:
        #     trajs = self.trajs
        #     trajs_next_states = self.trajs_next_states
        #     trajs_actions = self.trajs_actions
        #     trajs_rewards = self.trajs_rewards
        #     trajs_dones = self.trajs_dones
        #     trajs_lengths = self.trajs_lengths[:-1]
        # else:
        traj_indices = np.random.randint(0, self.size, size=batch_size)
        trajs = np.array(self.trajs, dtype=object)[traj_indices]
        trajs_next_states = np.array(self.trajs_next_states, dtype=object)[traj_indices]
        trajs_actions = np.array(self.trajs_actions, dtype=object)[traj_indices]
        trajs_rewards = np.array(self.trajs_rewards, dtype=object)[traj_indices]
        trajs_dones = np.array(self.trajs_dones, dtype=object)[traj_indices]
        trajs_lengths = np.array(self.trajs_lengths)[traj_indices]
        
        if type(trajs[0]) == np.ndarray:
            trajs = torch.Tensor(trajs.astype(np.float32)).type(torch.float)
            trajs_next_states = torch.Tensor(trajs_next_states.astype(np.float32)).type(torch.float)
            trajs_actions = torch.Tensor(trajs_actions.astype(np.float32)).type(torch.float)
            trajs_rewards = torch.Tensor(trajs_rewards.astype(np.float32)).type(torch.float)
            trajs_dones = torch.Tensor(trajs_dones.astype(np.float32)).type(torch.float)
            
        # Padding values
        min_length = np.min(trajs_lengths)
        max_length = min(np.max(trajs_lengths), 400)
        if min_length == max_length:
            index_to_clip = min_length
        else:
            index_to_clip = np.random.randint(min_length, max_length) 
        
        clipped_trajs_length = np.clip(trajs_lengths, 0, index_to_clip)
        last_step_ind = clipped_trajs_length - 1

        padded_trajs = torch.nn.utils.rnn.pad_sequence(trajs, batch_first=True, padding_side='right')
        padded_trajs_next_states = torch.nn.utils.rnn.pad_sequence(trajs_next_states, batch_first=True, padding_side='right')
        print(padded_trajs.shape)
        print(clipped_trajs_length)
        tensor_trajs = torch.nn.utils.rnn.pack_padded_sequence(padded_trajs, lengths=clipped_trajs_length, batch_first=True, enforce_sorted=False).to(self.device)
        tensor_trajs_next_states = torch.nn.utils.rnn.pack_padded_sequence(padded_trajs_next_states, lengths=clipped_trajs_length, batch_first=True, enforce_sorted=False).to(self.device)
        
        padded_trajs_actions = torch.nn.utils.rnn.pad_sequence(trajs_actions, batch_first=True, padding_side='right')
        padded_trajs_rewards = torch.nn.utils.rnn.pad_sequence(trajs_rewards, batch_first=True, padding_side='right')
        padded_trajs_dones = torch.nn.utils.rnn.pad_sequence(trajs_dones, batch_first=True, padding_side='right', )
        
        print(padded_trajs_actions.shape)
        print(padded_trajs_rewards.shape)
        print(torch.tensor(last_step_ind).unsqueeze(-1).expand(-1, padded_trajs_actions.shape[-1]).unsqueeze(1).shape)
        tensor_trajs_actions = torch.gather(padded_trajs_actions, dim=1, index=torch.tensor(last_step_ind).unsqueeze(-1).expand(-1, padded_trajs_actions.shape[-1]).unsqueeze(1)).squeeze(1).to(self.device)
        tensor_trajs_rewards = torch.gather(padded_trajs_rewards, dim=1, index=torch.tensor(last_step_ind).unsqueeze(-1)).to(self.device)
        tensor_trajs_dones = torch.gather(padded_trajs_dones, dim=1, index=torch.tensor(last_step_ind).unsqueeze(-1)).to(self.device)
        
        tensor_trajs_rewards = tensor_trajs_rewards.type(torch.float)
        tensor_trajs_actions = tensor_trajs_actions.type(torch.float)
        tensor_trajs_dones = tensor_trajs_dones.type(torch.float)
        
        return TrajData(tensor_trajs, tensor_trajs_next_states, tensor_trajs_actions, tensor_trajs_rewards, tensor_trajs_dones)
        