import torch.nn as nn
import torch
import numpy as np

class FK_LSTM(nn.LSTM):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 batch_first,
                 device):
        super(FK_LSTM, self).__init__(input_size,
                                      hidden_size,
                                      num_layers,
                                      batch_first=batch_first)
        self.device = device
        print("device of LSTM", self.device)
        # self.h0 = torch.zeros(self.num_layers, input_size, self.hidden_size).to(self.device)
        # self.c0 = torch.zeros(self.num_layers, input_size, self.hidden_size).to(self.device)
        self.logstd = nn.Parameter(torch.tensor([0.1, 0.1]).to(self.device)).to(self.device)

        self.dense_net = nn.Linear(hidden_size, 2).to(self.device)
    
    def forward(self, x, prob=False):
        # out, _ = super(FK_LSTM, self).forward(x, (self.h0, self.c0))
        out, _ = super(FK_LSTM, self).forward(x)
        out = torch.tanh(self.dense_net(out)) * np.pi/4
        if prob:
            distribution = torch.distributions.Normal(loc=out,
                                                      scale=torch.exp(self.logstd))
            out = distribution.rsample()
        return out
    