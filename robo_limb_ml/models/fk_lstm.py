import torch.nn as nn
import torch
import numpy as np

class FK_LSTM(nn.LSTM):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 batch_first,
                 batch_size,
                 output_size,
                 device,
                 domain_boundary=100):
        super(FK_LSTM, self).__init__(input_size,
                                      hidden_size,
                                      num_layers,
                                      batch_first=batch_first)
        self.device = device
        print("device of LSTM", self.device)
        self.h0 = torch.zeros(self.num_layers, input_size, batch_size, self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers, input_size, batch_size, self.hidden_size).to(self.device)
        self.logstd = nn.Parameter(torch.tensor(np.ones(output_size)*0.1).to(self.device)).to(self.device)
        self.activation = nn.Tanh()
        self.dense_net = nn.Linear(hidden_size, output_size).to(self.device)
        self.boundary = domain_boundary
    
    # Allows for stateful or stateless LTSM
    def forward(self, x, hn, cn, prob=False):
        # out, _ = super(FK_LSTM, self).forward(x, (self.h0, self.c0))
        out, (out_hn, out_cn) = super(FK_LSTM, self).forward(x, (hn, cn))
        out = torch.tanh(self.dense_net(out)) * self.boundary
        if prob:
            distribution = torch.distributions.Normal(loc=out,
                                                      scale=torch.exp(self.logstd))
            out = distribution.rsample()
        return out, out_hn, out_cn
    