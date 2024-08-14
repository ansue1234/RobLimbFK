import torch.nn as nn
import numpy as np

class QNet_MLP(nn.Module):
    def __init__(self, env=None, input_dim=None, output_dim=None):
        super().__init__()
        if input_dim is None:
            input_dim = np.array(env.single_observation_space.shape).prod()
        if output_dim is None:
            output_dim = env.single_action_space.n
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.network(x)
    

class QNet_LSTM(nn.Module):
    def __init__(self, env=None, input_dim=None, output_dim=None):
        super().__init__()
        if input_dim is None:
            input_dim = np.array(env.single_observation_space.shape).prod()
        if output_dim is None:
            output_dim = env.single_action_space.n
        self.lstm = nn.LSTM(input_dim, 1024, 3, batch_first=True)
        self.fc = nn.Linear(1024, output_dim)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        x = self.fc(x)[:, -1, :]
        return x, (h_n, c_n)