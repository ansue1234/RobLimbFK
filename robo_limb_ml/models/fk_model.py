import torch.nn as nn
import torch.distributions
import torch.nn.init as init
import numpy as np

class PsuedoFKModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_sizes: list, 
                 output_size: int,
                 distribution_name: str = 't',
                 distribution_params: dict = {}):
        super(PsuedoFKModel, self).__init__()

        # Define the model
        layers = [nn.Linear(input_size, hidden_sizes[0]), 
                  nn.ReLU()]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.model = nn.Sequential(*layers)

        # Initialize weights using kaiming_uniform and biases to 0 due to relu
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                init.constant_(layer.bias, 0)
        
        # 3 types of distribution to try out
        # 1. t-distribution params: df, std
        # 2. normal distribution: params: std
        # 3. uniform distribution: params: +-range
        self.distribution_params = distribution_params
        self.distribution_name = distribution_name


    def forward(self, x):
        # wrapping in between -pi/4 to pi/4 i.e. -45 to 45 degrees 
        x = self.model(x)
        x = x % np.pi/2 - np.pi/4
        return x
    
    def forward_prob(self, x):
        mean = self.forward(x)
        if self.distribution_name == 't':
            self.distribution = torch.distributions.StudentT(df=self.distribution_params['df'],
                                                             scale=self.distribution_params['std'],
                                                             loc=mean)
        elif self.distribution_name == 'normal':
            self.distribution = torch.distributions.Normal(loc=mean,
                                                           scale=self.distribution_params['std'])
        else:
            self.distribution = torch.distributions.Uniform(low=mean-self.distribution_params['range'],
                                                            high=mean+self.distribution_params['range'])
        # clamping results to -45 to 45 degrees
        sampled_angle = torch.clamp(self.distribution.sample(), -np.pi/4, np.pi/4)
        return sampled_angle

        
        