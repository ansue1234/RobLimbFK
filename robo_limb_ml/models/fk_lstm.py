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
                                      batch_first=batch_first,
                                      device=device)
        self.device = device
        print("device of LSTM", self.device)
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        self.logstd = nn.Parameter(torch.tensor(np.ones(output_size)*0.1).to(self.device)).to(self.device)
        self.activation = nn.Tanh()
        self.dense_net = nn.Linear(hidden_size, output_size).to(self.device)
        self.layer_norm = nn.LayerNorm(hidden_size).to(self.device)
        self.boundary = domain_boundary
    
    # Allows for stateful or stateless LTSM
    def forward(self, x, hn, cn, prob=False):
        # out, _ = super(FK_LSTM, self).forward(x, (self.h0, self.c0))
        # print(x.shape)
        out, (out_hn, out_cn) = super(FK_LSTM, self).forward(x, (hn, cn))
        out = self.layer_norm(out)
        out_cn = self.layer_norm(out_cn)
        out_hn = self.layer_norm(out_hn)
        out = self.dense_net(out)
        if prob:
            distribution = torch.distributions.Normal(loc=out,
                                                      scale=torch.exp(self.logstd))
            out = distribution.rsample().float()
        out = torch.tanh(out) * self.boundary
        out = out[:, -1: ,:]
        return out, out_hn, out_cn
    