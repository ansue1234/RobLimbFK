import torch
import torch.nn as nn

class FK_MLP(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 device,
                 domain_boundary=100):
        super(FK_MLP, self).__init__()
        self.device = device
        self.logstd = nn.Parameter(torch.tensor([0.1]*output_size).to(self.device))
        self.activation = nn.Tanh()
        layers = [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)]
        self.dense_net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            self.activation,
            *layers,
            nn.Linear(hidden_sizes[-1], output_size)
        ).to(self.device)
        self.boundary = domain_boundary
    
    def forward(self, x, prob=False):
        out = self.dense_net(x)
        if prob:
            distribution = torch.distributions.Normal(loc=out,
                                                      scale=torch.exp(self.logstd))
            out = distribution.rsample().float()
        out = torch.tanh(out) * self.boundary
        return out