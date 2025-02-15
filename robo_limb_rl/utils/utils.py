import torch 
import numpy as np
from typing import Tuple
from torch import Tensor
from torch import jit
from torch.nn.utils.rnn import PackedSequence


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

# only for single environment
class TrajData():
    def __init__(self, state_orig_obs, state_new_obs, next_state_orig_obs, next_state_new_obs, action, reward, done):
        self.observations = (state_orig_obs, state_new_obs)
        self.next_observations = (next_state_orig_obs, next_state_new_obs)
        self.actions = action
        self.rewards = reward
        self.dones = done

class TrajReplayBuffer():
    def __init__(self, max_size, device):
        self.max_size = max_size
        self.device = device
        self.size = 0
        self.trajs_orig_obs = []
        self.trajs_new_obs = []
        self.trajs_next_states_orig_obs = []
        self.trajs_next_states_new_obs = []
        self.trajs_actions = []
        self.trajs_rewards = []
        self.trajs_dones = []
        self.trajs_lengths = [0]
        self.trajs_info = []
        
        self.current_traj_orig_obs = []
        self.current_traj_new_obs = []
        self.current_traj_next_states_orig_obs = []
        self.current_traj_next_states_new_obs = []
        self.current_traj_actions = []
        self.current_traj_rewards = []
        self.current_traj_dones = []
        self.current_traj_info = []
    
    def add(self, state, next_state, action, reward, done, truncated, infos):
        # Do splicing here
        state = state[0]
        next_state = next_state[0]
        
        state_orig_obs = state[:6]
        state_new_obs = state[6:]
        next_state_orig_obs = next_state[:6]
        next_state_new_obs = next_state[6:]
        
        action = action[0]
        reward = reward[0]
        done = done[0]
        truncated = truncated[0]
        
        self.current_traj_orig_obs.append(state_orig_obs)
        self.current_traj_new_obs.append(state_new_obs)
        self.current_traj_next_states_orig_obs.append(next_state_orig_obs)
        self.current_traj_next_states_new_obs.append(next_state_new_obs)
        self.current_traj_actions.append(action)
        self.current_traj_rewards.append(reward)
        self.current_traj_dones.append(done)
        self.trajs_lengths[-1] += 1
        
        if done or truncated:
            
            self.trajs_orig_obs.append(torch.Tensor(np.array(self.current_traj_orig_obs)))
            self.trajs_new_obs.append(torch.Tensor(np.array(self.current_traj_new_obs)))
            self.trajs_next_states_orig_obs.append(torch.Tensor(np.array(self.current_traj_next_states_orig_obs)))
            self.trajs_next_states_new_obs.append(torch.Tensor(np.array(self.current_traj_next_states_new_obs)))
            self.trajs_actions.append(torch.Tensor(self.current_traj_actions))
            self.trajs_rewards.append(torch.Tensor(self.current_traj_rewards))
            self.trajs_dones.append(torch.Tensor(self.current_traj_dones))
            
            
            self.current_traj_orig_obs = []
            self.current_traj_new_obs = []
            self.current_traj_next_states_orig_obs = []
            self.current_traj_next_states_new_obs = []
            self.current_traj_actions = []
            self.current_traj_rewards = []
            self.current_traj_dones = []
            self.trajs_lengths.append(0)
            
            if self.size < self.max_size:
                self.size += 1
            else:
                self.size = self.max_size
                self.trajs_orig_obs.pop(0)
                self.trajs_new_obs.pop(0)
                self.trajs_next_states_orig_obs.pop(0)
                self.trajs_next_states_new_obs.pop(0)
                self.trajs_actions.pop(0)
                self.trajs_rewards.pop(0)
                self.trajs_dones.pop(0)
                self.trajs_lengths.pop(0)
                
    def pop(self):
        self.trajs_orig_obs.pop(0)
        self.trajs_new_obs.pop(0)
        self.trajs_next_states_orig_obs.pop(0)
        self.trajs_next_states_new_obs.pop(0)
        self.trajs_actions.pop(0)
        self.trajs_rewards.pop(0)
        self.trajs_dones.pop(0)
        if len(self.trajs_lengths) > 1:
            self.trajs_lengths.pop(0)
            self.size -= 1
        else:
            self.size = 0
    
    def sample(self, batch_size=1):
        first_traj = False
        if self.size < 1:
            first_traj = True
            self.size = 1
            self.trajs_orig_obs.append(torch.Tensor(self.current_traj_orig_obs))
            self.trajs_new_obs.append(torch.Tensor(self.current_traj_new_obs))
            self.trajs_next_states_orig_obs.append(torch.Tensor(self.current_traj_next_states_orig_obs))
            self.trajs_next_states_new_obs.append(torch.Tensor(self.current_traj_next_states_new_obs))
            self.trajs_actions.append(torch.Tensor(self.current_traj_actions))
            self.trajs_rewards.append(torch.Tensor(self.current_traj_rewards))
            self.trajs_dones.append(torch.Tensor(self.current_traj_dones))
        # if self.size < batch_size:
        #     trajs = self.trajs
        #     trajs_next_states = self.trajs_next_states
        #     trajs_actions = self.trajs_actions
        #     trajs_rewards = self.trajs_rewards
        #     trajs_dones = self.trajs_dones
        #     trajs_lengths = self.trajs_lengths[:-1]
        # else:
        traj_indices = np.random.randint(0, self.size, size=batch_size)
        trajs_orig_obs = np.array(self.trajs_orig_obs, dtype=object)[traj_indices]
        trajs_new_obs = np.array(self.trajs_new_obs, dtype=object)[traj_indices]
        trajs_next_states_orig_obs = np.array(self.trajs_next_states_orig_obs, dtype=object)[traj_indices]
        trajs_next_states_new_obs = np.array(self.trajs_next_states_new_obs, dtype=object)[traj_indices]
        trajs_actions = np.array(self.trajs_actions, dtype=object)[traj_indices]
        trajs_rewards = np.array(self.trajs_rewards, dtype=object)[traj_indices]
        trajs_dones = np.array(self.trajs_dones, dtype=object)[traj_indices]
        trajs_lengths = np.array(self.trajs_lengths)[traj_indices]
        
        if type(trajs_orig_obs[0]) == np.ndarray:
            trajs_orig_obs = torch.Tensor(trajs_orig_obs.astype(np.float32)).type(torch.float)
            trajs_new_obs = torch.Tensor(trajs_new_obs.astype(np.float32)).type(torch.float)
            trajs_next_states_new_obs = torch.Tensor(trajs_next_states_new_obs.astype(np.float32)).type(torch.float)
            trajs_next_states_orig_obs = torch.Tensor(trajs_next_states_orig_obs.astype(np.float32)).type(torch.float)
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

        padded_trajs_orig_obs = torch.nn.utils.rnn.pad_sequence(trajs_orig_obs, batch_first=True, padding_side='right')
        padded_trajs_new_obs = torch.nn.utils.rnn.pad_sequence(trajs_new_obs, batch_first=True, padding_side='right')
        padded_trajs_next_states_orig_obs = torch.nn.utils.rnn.pad_sequence(trajs_next_states_orig_obs, batch_first=True, padding_side='right')
        padded_trajs_next_states_new_obs = torch.nn.utils.rnn.pad_sequence(trajs_next_states_new_obs, batch_first=True, padding_side='right')
        
        tensor_trajs_orig_obs = torch.nn.utils.rnn.pack_padded_sequence(padded_trajs_orig_obs, lengths=clipped_trajs_length, batch_first=True, enforce_sorted=False).to(self.device)
        tensor_trajs_new_obs = torch.nn.utils.rnn.pack_padded_sequence(padded_trajs_new_obs, lengths=clipped_trajs_length, batch_first=True, enforce_sorted=False).to(self.device)
        tensor_trajs_next_states_orig_obs = torch.nn.utils.rnn.pack_padded_sequence(padded_trajs_next_states_orig_obs, lengths=clipped_trajs_length, batch_first=True, enforce_sorted=False).to(self.device)
        tensor_trajs_next_states_new_obs = torch.nn.utils.rnn.pack_padded_sequence(padded_trajs_next_states_new_obs, lengths=clipped_trajs_length, batch_first=True, enforce_sorted=False).to(self.device)
        
        padded_trajs_actions = torch.nn.utils.rnn.pad_sequence(trajs_actions, batch_first=True, padding_side='right')
        padded_trajs_rewards = torch.nn.utils.rnn.pad_sequence(trajs_rewards, batch_first=True, padding_side='right')
        padded_trajs_dones = torch.nn.utils.rnn.pad_sequence(trajs_dones, batch_first=True, padding_side='right', )
        
        # print(torch.tensor(last_step_ind).unsqueeze(-1).expand(-1, padded_trajs_actions.shape[-1]).unsqueeze(1).shape)
        tensor_trajs_actions = torch.gather(padded_trajs_actions, dim=1, index=torch.tensor(last_step_ind).unsqueeze(-1).expand(-1, padded_trajs_actions.shape[-1]).unsqueeze(1)).squeeze(1).to(self.device)
        tensor_trajs_rewards = torch.gather(padded_trajs_rewards, dim=1, index=torch.tensor(last_step_ind).unsqueeze(-1)).to(self.device)
        tensor_trajs_dones = torch.gather(padded_trajs_dones, dim=1, index=torch.tensor(last_step_ind).unsqueeze(-1)).to(self.device)
        
        tensor_trajs_rewards = tensor_trajs_rewards.type(torch.float)
        tensor_trajs_actions = tensor_trajs_actions.type(torch.float)
        tensor_trajs_dones = tensor_trajs_dones.type(torch.float)
        
        # handle first traj
        if first_traj:
            self.pop()
        # return TrajData(tensor_trajs, tensor_trajs_next_states, tensor_trajs_actions, tensor_trajs_rewards, tensor_trajs_dones)
        return TrajData(tensor_trajs_orig_obs, 
                        tensor_trajs_new_obs, 
                        tensor_trajs_next_states_orig_obs, 
                        tensor_trajs_next_states_new_obs, 
                        tensor_trajs_actions, 
                        tensor_trajs_rewards, 
                        tensor_trajs_dones)


# Dealing with packed sequence and padded sequences
# https://discuss.pytorch.org/t/get-each-sequences-last-item-from-packed-sequence/41118/8


@jit.script
def sorted_lengths(pack: PackedSequence) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(
        pack.batch_sizes[0],
        dtype=pack.batch_sizes.dtype,
        device=pack.batch_sizes.device,
    )
    lengths = ((indices + 1)[:, None] <= pack.batch_sizes[None, :]).long().sum(dim=1)
    return lengths, indices


@jit.script
def sorted_first_indices(pack: PackedSequence) -> torch.Tensor:
    return torch.arange(
        pack.batch_sizes[0],
        dtype=pack.batch_sizes.dtype,
        device=pack.batch_sizes.device,
    )


@jit.script
def sorted_last_indices(pack: PackedSequence) -> torch.Tensor:
    lengths, indices = sorted_lengths(pack)
    cum_batch_sizes = torch.cat([
        pack.batch_sizes.new_zeros((2,)),
        torch.cumsum(pack.batch_sizes, dim=0),
    ], dim=0)
    return cum_batch_sizes[lengths] + indices


@jit.script
def first_items(pack: PackedSequence, unsort: bool) -> torch.Tensor:
    if unsort and pack.unsorted_indices is not None:
        return pack.data[pack.unsorted_indices]
    else:
        return pack.data[:pack.batch_sizes[0]]


@jit.script
def last_items(pack: PackedSequence, unsort: bool) -> torch.Tensor:
    indices = sorted_last_indices(pack=pack)
    if unsort and pack.unsorted_indices is not None:
        indices = indices[pack.unsorted_indices]
    return pack.data[indices]

def get_last_items(x):
    if type(x) is torch.Tensor:
        return x[:, -1, :].unsqueeze(1)
    else:
        return last_items(x, unsort=True).unsqueeze(1)