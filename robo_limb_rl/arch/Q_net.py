import torch.nn as nn
import torch
import numpy as np

class QNet_MLP(nn.Module):
    def __init__(self, env=None, input_dim=None, output_dim=None, gamma=0.99, lam=0.01, reward_type='reg'):
        super().__init__()
        if input_dim is None:
            input_dim = np.array(env.single_observation_space.shape).prod()
        if output_dim is None:
            output_dim = env.single_action_space.n
        if reward_type == 'reg':
            self.gamma = gamma
        else:
            self.lam = lam
        self.reward_type = reward_type
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        # self.network = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, output_dim),
        # )

    def forward(self, x):
        out = self.network(x)
        if self.reward_type == 'reg':
            out = (1/(1-self.gamma))*torch.tanh(out)
        elif self.reward_type == 'e_exp':
            out = np.exp(-(1/(1-self.lam))) * torch.sigmoid(out)
        else:
            out = np.power(1+self.lam, -(1/(1-self.lam))) * torch.sigmoid(out)
        return out
    
class QNet_LSTM(nn.Module):
    def __init__(self, env=None, input_dim=None, output_dim=None, gamma=0.99, lam=0.01, reward_type='reg'):
        super().__init__()
        if input_dim is None:
            input_dim = np.array(env.single_observation_space.shape).prod()
        if output_dim is None:
            output_dim = env.single_action_space.n
        if reward_type == 'reg':
            self.gamma = gamma
        else:
            self.lam = lam
        self.reward_type = reward_type
        self.lstm = nn.LSTM(input_dim, 1024, 3, batch_first=True)
        self.fc = nn.Linear(1024, output_dim)

    def forward(self, x, hidden=None):
        if hidden:
            out, (h_n, c_n) = self.lstm(x, hidden)
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(out)[:, -1, :]
        if self.reward_type == 'reg':
            out = (1/(1-self.gamma))*torch.tanh(out)
        elif self.reward_type == 'e_exp':
            out = np.exp(-(1/(1-self.lam))) * torch.sigmoid(out)
        else:
            out = np.power(1+self.lam, -(1/(1-self.lam))) * torch.sigmoid(out)
        return out, (h_n, c_n)