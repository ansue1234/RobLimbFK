import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
from robo_limb_ml.models.fk_seq2seq import FK_SEQ2SEQ
from robo_limb_rl.utils.utils import get_last_items, last_items

class Seq2SeqEncoderOnly(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_size, pretrained_model=None, device='cpu'):
        super(Seq2SeqEncoderOnly, self).__init__()
        self.device = device
        seq2seq_net = FK_SEQ2SEQ(input_size=input_dim,
                                 embedding_size=hidden_dim,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 batch_size=batch_size,
                                 output_size=4,
                                 device=self.device,
                                 encoder_type="LSTM",
                                 decoder_type="LSTM",
                                 domain_boundary=100,
                                 pred_len=1,
                                 attention=False,
                                 teacher_forcing_ratio=0.75)
        if pretrained_model is not None:
            # Load the entire model from the checkpoint
            loaded_model = torch.load(pretrained_model, map_location=self.device)
            seq2seq_net.load_state_dict(loaded_model)

        self.encoder = seq2seq_net.encoder
        # Move the encoder to the specified device
        self.encoder.to(self.device)

    def forward(self, x, hidden):
        # Ensure input is on the correct device
        x = x.to(self.device)
        # Forward pass through the encoder
        out, hidden = self.encoder(x, hidden)
        out = get_last_items(out)
        return out, hidden

class Seq2SeqFullHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_size, pretrained_model=None, device='cpu'):
        super(Seq2SeqFullHead, self).__init__()
        self.device = device
        seq2seq_net = FK_SEQ2SEQ(input_size=input_dim,
                                     embedding_size=hidden_dim,
                                     num_layers=num_layers,
                                     batch_first=True,
                                     batch_size=batch_size,
                                     output_size=4,
                                     device=self.device,
                                     encoder_type="LSTM",
                                     decoder_type="LSTM",
                                     domain_boundary=100,
                                     pred_len=1,
                                     attention=False,
                                     teacher_forcing_ratio=0.0)
        
        if pretrained_model is not None:
            loaded_model = torch.load(pretrained_model, map_location=self.device)
            seq2seq_net.load_state_dict(loaded_model)
            # Load components from the pretrained model
        self.encoder = seq2seq_net.encoder
        self.decoder = seq2seq_net.decoder.decoder  # Access the underlying LSTM/RNN module
        # self.decoder_input_size = seq2seq_net.decoder.decoder.input_size
        
        # Ensure components are on the correct device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.seq2seq_net = seq2seq_net

    def forward(self, x, hidden):
        # print("main forward", x.shape)
        # encoder_out, encoder_hidden = self.encoder(x, hidden)
        # decoder_input = get_last_items(x)
        # decoder_hidden = encoder_hidden
        # decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_out)
        # decoder_out = get_last_items(decoder_out)
        # return decoder_out, encoder_hidden
        
        return self.seq2seq_net(x, None, hidden)


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(MLPHead, self).__init__()
        self.nn = nn.Sequential(
                    nn.Linear(input_dim*seq_len, hidden_dim*2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim*2, hidden_dim*2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim*2, hidden_dim))
        self.input_dim = input_dim
        self.seq_len = seq_len

    def forward(self, x, hidden=None):
        x = torch.flatten(x.clone(), start_dim=1)
        if x.shape[1] < self.input_dim*self.seq_len:
            pad_length = self.input_dim*self.seq_len - x.shape[1]
            x = F.pad(x, (0, pad_length), "constant", 0)
        return self.nn(x), None

class EmptyHead(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(EmptyHead, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

    def forward(self, x, hidden=None):
        x = torch.flatten(x.clone(), start_dim=1)
        if x.shape[1] < self.input_dim*self.seq_len:
            pad_length = self.input_dim*self.seq_len - x.shape[1]
            x = F.pad(x, (0, pad_length), "constant", 0)
        return x, None

class QNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512) # Original 256
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SACActor(nn.Module):
    def __init__(self, input_dim, action_space):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512) # Original 256
        self.fc2 = nn.Linear(512, 512)
        self.fc_mean = nn.Linear(512, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(512, np.prod(action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        )
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

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

class TD3Actor(nn.Module):
    def __init__(self, input_dim, action_space):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class RLAgent(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 head_type='mlp',
                 agent='SAC',
                 hidden_dim=256,
                 num_layers=1,
                 batch_size=1,
                 seq_len=100,
                 pretrained_model=None,
                 freeze_head=False,
                 device='cpu'):
        super(RLAgent, self).__init__()
        self.device = device
        self.head_type = head_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.state_dim = 6 # x, y, vel x, vel y, acc x, acc y
        # Initialize the shared head
        if head_type == 'seq2seq_encoder':
            self.head = Seq2SeqEncoderOnly(self.state_dim, hidden_dim, num_layers, batch_size, pretrained_model, device)
            self.output_dim = hidden_dim
        elif head_type == 'seq2seq_full':
            self.head = Seq2SeqFullHead(self.state_dim, hidden_dim, num_layers, batch_size, pretrained_model, device)
            self.output_dim = 4
            # self.hidden_dim = 4
        elif head_type == 'mlp':
            self.head = MLPHead(self.state_dim, hidden_dim, self.seq_len)
            self.output_dim = hidden_dim
        else:
            self.head = EmptyHead(observation_space.shape[0], self.seq_len)
            self.output_dim = observation_space.shape[0]*self.seq_len

        # Initialize actor and include skip connections
        if agent == 'SAC':
            self.actor = SACActor(self.output_dim + np.prod(observation_space.shape), action_space)
        elif agent == 'TD3':
            self.actor = TD3Actor(self.output_dim + np.prod(observation_space.shape), action_space)
        
        if head_type == 'seq2seq_encoder' or head_type == 'seq2seq_full':
            self.layer_norm = nn.LayerNorm(self.output_dim)
        
        # Initialize critics
        self.critic1 = QNetwork(self.output_dim + np.prod(observation_space.shape) + np.prod(action_space.shape))
        self.critic2 = QNetwork(self.output_dim + np.prod(observation_space.shape) + np.prod(action_space.shape))
        self.freeze_head = freeze_head

    def _get_features(self, x, hidden=None):
        # No context because each batch contains whole trajectory
        if type(x) is not torch.Tensor:
            batch_size = x[0].shape[0]
        else:
            batch_size = x.shape[0]
        
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)) if hidden is None else hidden
        if self.freeze_head:
            with torch.no_grad():
                features, h = self.head(x, hidden)
                if self.head_type == 'seq2seq_encoder' or self.head_type == 'seq2seq_full':
                    features = self.layer_norm(features.squeeze(1))
                return features, h
        else:
            features, h = self.head(x, hidden)
            if self.head_type == 'seq2seq_encoder' or self.head_type == 'seq2seq_full':
                features = self.layer_norm(features.squeeze(1))
            return features, h

    def get_action(self, x, hidden=None):
        # Ensure input is a tensor on the correct device
        # print(x)
        # if not isinstance(x, torch.Tensor):
        #     x = torch.FloatTensor(x).to(self.device)
        # Add necessary dimensions for sequential heads
        
        # Breaking into obs and power_goal
        if type(x) == tuple:
            obs, power_goal = x[0], x[1]
            power_goal = last_items(power_goal, unsort=True)
            # print("power_goal", power_goal)
        else:
            obs = x[:,:,:self.state_dim]
            power_goal = x[:,-1, self.state_dim:]
            if type(x) is not torch.Tensor:
                obs = torch.FloatTensor(obs).to(self.device)
                power_goal = torch.FloatTensor(power_goal).to(self.device)

        # Get features and update hidden state
        features, h = self._get_features(obs, hidden)
        # print("features", torch.mean(features), torch.max(features), torch.min(features), torch.std(features))
        # print("power_goal", power_goal)
        features = torch.cat((features, obs[:, -1, :], power_goal), dim=1)
        
        # Get action from actor
        action, log_prob, mean = self.actor.get_action(features)
        return action, log_prob, mean, h

    def forward_critic(self, x, a, hidden=None):
        # Process state through the shared head
        if type(x) == tuple:
            obs, power_goal = x[0], x[1]
            power_goal = last_items(power_goal, unsort=True)
        else:
            obs = x[:,:,:self.state_dim]
            power_goal = x[:, -1, self.state_dim:]
            if type(x) is not torch.Tensor:
                obs = torch.FloatTensor(obs).to(self.device)
                power_goal = torch.FloatTensor(power_goal).to(self.device)
                a = torch.FloatTensor(a).to(self.device)
            
        features, h = self._get_features(obs, hidden)
        features = torch.cat((features, obs[:, -1, :], power_goal), dim=1)
        # Forward through critics
        q1 = self.critic1(features, a)
        q2 = self.critic2(features, a)
        return q1, q2, h

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self,
                 obs_space_dim,
                 act_space_dim,
                 head_type='seq2seq_encoder',
                 hidden_dim=512,
                 num_layers=1,
                 batch_size=1,
                 pretrained_model=None,
                 freeze_head=False,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.head_type = head_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.state_dim = 6 # x, y, vel x, vel y, acc x, acc y
        self.obs_space_dim = obs_space_dim
        self.act_space_dim = act_space_dim
        self.freeze_head = freeze_head
        self.device = device
        self.pretrained_model = pretrained_model
        # Initialize the shared head
        if head_type == 'seq2seq_encoder':
            self.head = Seq2SeqEncoderOnly(self.state_dim, hidden_dim, num_layers, batch_size, pretrained_model, device)
        elif head_type == 'seq2seq_full':
            self.head = Seq2SeqFullHead(self.state_dim, hidden_dim, num_layers, batch_size, pretrained_model, device)
        elif head_type == 'mlp':
            self.head = MLPHead(self.obs_space_dim, hidden_dim)
        else:
            self.head = EmptyHead()
            self.hidden_dim = obs_space_dim
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim + self.obs_space_dim - self.state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim + self.obs_space_dim - self.state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, act_space_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_space_dim))
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def get_value(self, x, hidden=None):
        inputs, h = self._prep_input(x, hidden)
        return self.critic(inputs), h

    def get_action_and_value(self, x, action=None, hidden=None):
        inputs, h = self._prep_input(x, hidden)
        action_mean = self.actor_mean(inputs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(inputs), h

    def _get_features(self, x, hidden=None):
        # No context because each batch contains window of states 
        batch_size = x.shape[0]
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)) if hidden is None else hidden
        
        if self.freeze_head:
            with torch.no_grad():
                features, h = self.head(x, hidden)
                features = self.layer_norm(features.squeeze(1))
                return features, h
        else:
            features, h = self.head(x, hidden)
            features = self.layer_norm(features.squeeze(1))
            return features, h
    
    def _prep_input(self, x, hidden=None):
        x = x.to(self.device)
        obs = x[:,:,:self.state_dim]
        power_goal = x[:,-1, self.state_dim:]
        
        features, h = self._get_features(obs, hidden)
        inputs = torch.cat((features, power_goal), dim=1)
        return inputs, h