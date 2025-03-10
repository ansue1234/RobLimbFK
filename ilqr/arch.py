import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from robo_limb_ml.models.fk_seq2seq import FK_SEQ2SEQ

class LimbModel(nn.Module):
    def __init__(self, model_path, input_size, hidden_size, num_layers, attention, device, debugger=None):
        super(LimbModel, self).__init__()
        
        self.device = device
        self.hidden = (torch.zeros(num_layers, 1, hidden_size).to(device=self.device),
                       torch.zeros(num_layers, 1, hidden_size).to(device=self.device))
        self.model = FK_SEQ2SEQ(input_size=input_size,
                                    embedding_size=hidden_size,
                                    num_layers=num_layers,
                                    batch_size=1,
                                    output_size=4,
                                    device=self.device,
                                    batch_first=True,
                                    encoder_type='LSTM',
                                    decoder_type='LSTM',
                                    attention=attention,
                                    pred_len=1,
                                    teacher_forcing_ratio=0.0).to(device=self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.encoder.h0, self.model.encoder.c0 = self.hidden
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.debugger = debugger
        self.t = 0

    def forward(self, x, u, hidden, grad=True):
        u = torch.tensor(u, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        self.debugger.get_logger().info("x, u:" + str(x.shape) + str(u.shape))
        x_u = torch.cat((x, u), 0).unsqueeze(0).unsqueeze(0)
        # x = torch.tensor(x, dtype=torch.float32)
        if grad:
            delta_state, hidden = self.model(x_u, None, hidden, mode='test')
            output = x + delta_state
            self.t += 1
            if self.t % 100 == 0:
                hidden = (hidden[0].detach(), hidden[1].detach())
        else:
            with torch.no_grad():
                delta_state, hidden = self.model(x_u, None, hidden, mode='test')
                output = x + delta_state
        return output, hidden