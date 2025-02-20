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
    def __init__(self, state_orig_obs, state_new_obs, next_state_orig_obs, next_state_new_obs, action, reward, done, lengths):
        self.observations = (state_orig_obs, state_new_obs)
        self.next_observations = (next_state_orig_obs, next_state_new_obs)
        self.actions = action
        self.rewards = reward
        self.dones = done
        self.lengths = lengths
# only for single environment 1d vector for observations
class TrajReplayBuffer():
    def __init__(self, 
                 max_size, 
                 original_obs_space_size,
                 new_obs_space_size,
                 action_space_size,
                 device, 
                 min_seq_len=50,
                 max_seq_len=200):
        self.max_size = max_size
        self.device = device
        self.size = 0  # number of finished trajectories stored
        self.original_obs_space_size = original_obs_space_size
        self.new_obs_space_size = new_obs_space_size
        self.action_space_size = action_space_size
        
        # Pointer for where to write the next finished trajectory.
        self.buffer_pointer = 0
        # Pointer for the current step within the ongoing trajectory.
        self.traj_pos_pointer = 0

        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

        # Pre-allocated buffers for full trajectories.
        self.trajs_orig_obs = torch.zeros((max_size, max_seq_len, original_obs_space_size),
                                          dtype=torch.float32, device=device)
        self.trajs_new_obs = torch.zeros((max_size, max_seq_len, new_obs_space_size),
                                         dtype=torch.float32, device=device)
        self.trajs_next_states_orig_obs = torch.zeros((max_size, max_seq_len, original_obs_space_size),
                                                      dtype=torch.float32, device=device)
        self.trajs_next_states_new_obs = torch.zeros((max_size, max_seq_len, new_obs_space_size),
                                                     dtype=torch.float32, device=device)
        self.trajs_actions = torch.zeros((max_size, max_seq_len, action_space_size),
                                         dtype=torch.float32, device=device)
        self.trajs_rewards = torch.zeros((max_size, max_seq_len),
                                         dtype=torch.float32, device=device)
        self.trajs_dones = torch.zeros((max_size, max_seq_len),
                                       dtype=torch.float32, device=device)
        # This tensor stores the actual trajectory length for each stored trajectory.
        self.trajs_lengths = torch.zeros((max_size,), dtype=torch.int32, device=device)

    def add(self, state, next_state, action, reward, done, truncated, infos):
        """
        Writes the incoming step directly into the appropriate row and column of the replay buffer.
        When a terminal (or truncated) signal is received or when max_seq_len is reached,
        the current trajectory is considered finished.
        """
        # For a single environment, extract the first element.
        state = state[0]
        next_state = next_state[0]

        # Split the observation into its “original” and “new” parts.
        state_orig_obs = state[:6]
        state_new_obs = state[6:]
        next_state_orig_obs = next_state[:6]
        next_state_new_obs = next_state[6:]

        action = action[0]
        reward = reward[0]
        done = done[0]
        truncated = truncated[0]

        pos = self.traj_pos_pointer  # current step index in trajectory
        bp = self.buffer_pointer     # current trajectory index in buffer

        # Write the data directly into the main buffers.
        self.trajs_orig_obs[bp, pos] = torch.tensor(state_orig_obs, dtype=torch.float32, device=self.device)
        self.trajs_new_obs[bp, pos] = torch.tensor(state_new_obs, dtype=torch.float32, device=self.device)
        self.trajs_next_states_orig_obs[bp, pos] = torch.tensor(next_state_orig_obs, dtype=torch.float32, device=self.device)
        self.trajs_next_states_new_obs[bp, pos] = torch.tensor(next_state_new_obs, dtype=torch.float32, device=self.device)
        self.trajs_actions[bp, pos] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.trajs_rewards[bp, pos] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.trajs_dones[bp, pos] = torch.tensor(done, dtype=torch.float32, device=self.device)

        self.traj_pos_pointer += 1

        # If the trajectory has ended (or reached the maximum sequence length), mark it as finished.
        if done or truncated or self.traj_pos_pointer >= self.max_seq_len:
            length = self.traj_pos_pointer  # recorded trajectory length
            self.trajs_lengths[bp] = length

            # Update buffer pointer in a circular fashion.
            self.buffer_pointer = (self.buffer_pointer + 1) % self.max_size
            # Set entry at buffer_pointer in buffer to zeros
            self.trajs_orig_obs[self.buffer_pointer] = torch.zeros((self.max_seq_len, self.original_obs_space_size), dtype=torch.float32, device=self.device)
            self.trajs_new_obs[self.buffer_pointer] = torch.zeros((self.max_seq_len, self.new_obs_space_size), dtype=torch.float32, device=self.device)
            self.trajs_next_states_orig_obs[self.buffer_pointer] = torch.zeros((self.max_seq_len, self.original_obs_space_size), dtype=torch.float32, device=self.device)
            self.trajs_next_states_new_obs[self.buffer_pointer] = torch.zeros((self.max_seq_len, self.new_obs_space_size), dtype=torch.float32, device=self.device)
            self.trajs_actions[self.buffer_pointer] = torch.zeros((self.max_seq_len, self.action_space_size), dtype=torch.float32, device=self.device)
            self.trajs_rewards[self.buffer_pointer] = torch.zeros((self.max_seq_len), dtype=torch.float32, device=self.device)
            self.trajs_dones[self.buffer_pointer] = torch.zeros((self.max_seq_len), dtype=torch.float32, device=self.device)
            
            if self.size < self.max_size:
                self.size += 1

            # Reset the step pointer for the new trajectory.
            self.traj_pos_pointer = 0
    
    def sample(self, batch_size=1):
        """
        Samples a batch of trajectories from the replay buffer.
        """
        sampled_indices = np.random.choice(self.size, batch_size, replace=True)
        trajs_lengths = self.trajs_lengths[sampled_indices]
        
        max_length = int(trajs_lengths.max().item())
        min_length = int(trajs_lengths.min().item())
        
        if min_length < max_length and max_length > 0:
            index_to_clip_end = np.random.randint(min_length, max_length)
        elif max_length > self.min_seq_len:
            index_to_clip_end = np.random.randint(self.min_seq_len, max_length)
        else:
            index_to_clip_end = max_length
            
        index_to_clip_end = self.max_seq_len    
        # index_to_clip_end = np.random.randint(min_length, max_length) if min_length < max_length elif  np.random.randint(self.min_seq_len, max_length)
        index_to_clip_head = np.random.randint(0, index_to_clip_end - self.min_seq_len) if index_to_clip_end - self.min_seq_len < min_length and index_to_clip_end - self.min_seq_len > 0 else 0
        # disabling random to help with stable training
        index_to_clip_head = 0
        padded_trajs_orig_obs = self.trajs_orig_obs[sampled_indices][:, index_to_clip_head:index_to_clip_end]
        padded_trajs_new_obs = self.trajs_new_obs[sampled_indices][:, index_to_clip_head:index_to_clip_end]
        padded_trajs_next_states_orig_obs = self.trajs_next_states_orig_obs[sampled_indices][:, index_to_clip_head:index_to_clip_end]
        padded_trajs_next_states_new_obs = self.trajs_next_states_new_obs[sampled_indices][:, index_to_clip_head:index_to_clip_end]
        padded_trajs_actions = self.trajs_actions[sampled_indices][:, index_to_clip_head:index_to_clip_end]
        padded_trajs_rewards = self.trajs_rewards[sampled_indices][:, index_to_clip_head:index_to_clip_end]
        padded_trajs_dones = self.trajs_dones[sampled_indices][:, index_to_clip_head:index_to_clip_end]
        # print("Padded Trajs Length", padded_trajs_orig_obs.shape)
        
        clipped_trajs_length = torch.clamp(trajs_lengths, max=index_to_clip_end) - index_to_clip_head
        # print("Clipped Trajs Length", clipped_trajs_length)
        # print("max_length", max_length)
        # print("min_length", min_length)
        # print("index_to_clip_end", index_to_clip_end)
        # print("index_to_clip_head", index_to_clip_head)
        # print(clipped_trajs_length)
        clipped_trajs_length = clipped_trajs_length.to(dtype=torch.int32, device='cpu')
        
        last_step_ind = (clipped_trajs_length - 1).to(dtype=torch.int64, device=self.device)
        
        tensor_trajs_orig_obs = torch.nn.utils.rnn.pack_padded_sequence(padded_trajs_orig_obs, lengths=clipped_trajs_length, batch_first=True, enforce_sorted=False).to(self.device)
        tensor_trajs_new_obs = torch.nn.utils.rnn.pack_padded_sequence(padded_trajs_new_obs, lengths=clipped_trajs_length, batch_first=True, enforce_sorted=False).to(self.device)
        tensor_trajs_next_states_orig_obs = torch.nn.utils.rnn.pack_padded_sequence(padded_trajs_next_states_orig_obs, lengths=clipped_trajs_length, batch_first=True, enforce_sorted=False).to(self.device)
        tensor_trajs_next_states_new_obs = torch.nn.utils.rnn.pack_padded_sequence(padded_trajs_next_states_new_obs, lengths=clipped_trajs_length, batch_first=True, enforce_sorted=False).to(self.device)
        
        # print("sampled actions", padded_trajs_actions)
        # print(torch.tensor(last_step_ind).unsqueeze(-1).expand(-1, padded_trajs_actions.shape[-1]).unsqueeze(1).shape)
        tensor_trajs_actions = torch.gather(padded_trajs_actions, dim=1, index=last_step_ind.unsqueeze(-1).expand(-1, padded_trajs_actions.shape[-1]).unsqueeze(1)).squeeze(1).to(self.device)
        tensor_trajs_rewards = torch.gather(padded_trajs_rewards, dim=1, index=last_step_ind.unsqueeze(-1)).to(self.device)
        tensor_trajs_dones = torch.gather(padded_trajs_dones, dim=1, index=last_step_ind.unsqueeze(-1)).to(self.device)
        
        # print("picked actions", tensor_trajs_actions)
        
        tensor_trajs_rewards = tensor_trajs_rewards.type(torch.float)
        tensor_trajs_actions = tensor_trajs_actions.type(torch.float)
        tensor_trajs_dones = tensor_trajs_dones.type(torch.float)
        
        # return TrajData(tensor_trajs, tensor_trajs_next_states, tensor_trajs_actions, tensor_trajs_rewards, tensor_trajs_dones)
        return TrajData(tensor_trajs_orig_obs, 
                        tensor_trajs_new_obs, 
                        tensor_trajs_next_states_orig_obs, 
                        tensor_trajs_next_states_new_obs, 
                        tensor_trajs_actions, 
                        tensor_trajs_rewards, 
                        tensor_trajs_dones,
                        clipped_trajs_length)


# Dealing with packed sequence and padded sequences
# https://discuss.pytorch.org/t/get-each-sequences-last-item-from-packed-sequence/41118/8


@jit.script
def sorted_lengths(pack: PackedSequence) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(
        pack.batch_sizes[0],
        dtype=pack.batch_sizes.dtype,
        device=pack.data.device,
    )
    lengths = ((indices + 1)[:, None] <= pack.batch_sizes[None, :].to(device=pack.data.device)).long().sum(dim=1)
    return lengths, indices


@jit.script
def sorted_first_indices(pack: PackedSequence) -> torch.Tensor:
    return torch.arange(
        pack.batch_sizes[0],
        dtype=pack.batch_sizes.dtype,
        device=pack.data.device,
    )


@jit.script
def sorted_last_indices(pack: PackedSequence) -> torch.Tensor:
    lengths, indices = sorted_lengths(pack)
    cum_batch_sizes = torch.cat([
        pack.batch_sizes.new_zeros((2,)),
        torch.cumsum(pack.batch_sizes, dim=0),
    ], dim=0).to(device=pack.data.device)
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