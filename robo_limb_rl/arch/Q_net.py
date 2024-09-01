import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

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
        # self.network = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, output_dim),
        # )
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

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

class SoftQNetwork(nn.Module):
    def __init__(self, env=None, input_dim=None, output_dim=None, gamma=0.99, lam=0.01, reward_type='reg'):
        super().__init__()
        if input_dim is None:
            input_dim = np.array(env.single_observation_space.shape).prod()
        if output_dim is None:
            output_dim = np.prod(env.single_action_space.shape)
        if reward_type == 'reg':
            self.gamma = gamma
        else:
            self.lam = lam
        self.reward_type = reward_type
        self.fc1 = nn.Linear(input_dim + output_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        if self.reward_type == 'reg':
            x = (1/(1-self.gamma))*torch.tanh(x)
        elif self.reward_type == 'e_exp':
            x = np.exp(-(1/(1-self.lam))) * torch.sigmoid(x)
        else:
            x = np.power(1+self.lam, -(1/(1-self.lam))) * torch.sigmoid(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env=None, input_dim=None, output_dim=None):
        super().__init__()
        if input_dim is None:
            input_dim = np.array(env.single_observation_space.shape).prod()
        if output_dim is None:
            output_dim = np.prod(env.single_action_space.shape)
      
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, output_dim)
        self.fc_logstd = nn.Linear(256, output_dim)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
