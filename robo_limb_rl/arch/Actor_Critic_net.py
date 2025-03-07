import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
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
                                     teacher_forcing_ratio=0.75)
        
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

    def forward(self, x, hidden):
        # print("main forward", x.shape)
        encoder_out, encoder_hidden = self.encoder(x, hidden)
        decoder_input = get_last_items(x)
        decoder_hidden = encoder_hidden
        decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_out)
        decoder_out = get_last_items(decoder_out)
        return decoder_out, encoder_hidden


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPHead, self).__init__()
        self.nn = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, hidden=None):
        return self.nn(x), None

class EmptyHead(nn.Module):
    def __init__(self):
        super(EmptyHead, self).__init__()

    def forward(self, x, hidden=None):
        return x, None

class QNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128) # Original 256
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SACActor(nn.Module):
    def __init__(self, input_dim, action_space):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128) # Original 256
        self.fc2 = nn.Linear(128, 128)
        self.fc_mean = nn.Linear(128, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(128, np.prod(action_space.shape))
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
                 pretrained_model=None,
                 freeze_head=False,
                 device='cpu'):
        super(RLAgent, self).__init__()
        self.device = device
        self.head_type = head_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.state_dim = 6 
        # Initialize the shared head
        if head_type == 'seq2seq_encoder':
            self.head = Seq2SeqEncoderOnly(6, hidden_dim, num_layers, batch_size, pretrained_model, device)
        elif head_type == 'seq2seq_full':
            self.head = Seq2SeqFullHead(6, hidden_dim, num_layers, batch_size, pretrained_model, device)
        elif head_type == 'mlp':
            self.head = MLPHead(np.prod(observation_space.shape), hidden_dim)
        else:
            self.head = EmptyHead()
            hidden_dim = np.prod(observation_space.shape)

        # Initialize actor
        if agent == 'SAC':
            self.actor = SACActor(hidden_dim + np.prod(observation_space.shape) - self.state_dim, action_space)
        elif agent == 'TD3':
            self.actor = TD3Actor(hidden_dim + np.prod(observation_space.shape) - self.state_dim, action_space)
        self.actor_layer_norm = nn.LayerNorm(self.hidden_dim)

        # Initialize critics
        self.critic1 = QNetwork(hidden_dim + np.prod(observation_space.shape) - self.state_dim + np.prod(action_space.shape))
        self.critic2 = QNetwork(hidden_dim + np.prod(observation_space.shape) - self.state_dim + np.prod(action_space.shape))
        self.freeze_head = freeze_head
        self.critic_layer_norm = nn.LayerNorm(self.hidden_dim)

    def _get_features(self, x):
        # No context because each batch contains whole trajectory
        # dealing with padded sequence
        if type(x) is not torch.Tensor:
            batch_size = x[0].shape[0]
        else:
            batch_size = x.shape[0]
        
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))
        if self.freeze_head:
            with torch.no_grad():
                return self.head(x, hidden)
        else:
            return self.head(x, hidden)

    def get_action(self, x):
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

        # Get features and update hidden state
        features, _ = self._get_features(obs)
        features = features.squeeze(1)
        features = self.actor_layer_norm(features)
        
        features = torch.cat((features, power_goal), dim=1)
        # Get action from actor
        action, log_prob, mean = self.actor.get_action(features)
        return action, log_prob, mean

    def forward_critic(self, x, a):
        # Process state through the shared head
        if type(x) == tuple:
            obs, power_goal = x[0], x[1]
            power_goal = last_items(power_goal, unsort=True)
        else:
            obs = x[:,:,:self.state_dim]
            power_goal = x[:,-1,self.state_dim:]
            
        features, _ = self._get_features(obs)
        features = features.squeeze(1)
        features = self.critic_layer_norm(features)
        # unpacking power goal
        
        
        features = torch.cat((features, power_goal), dim=1)
        # Forward through critics
        q1 = self.critic1(features, a)
        q2 = self.critic2(features, a)
        return q1, q2