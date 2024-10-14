import torch
import torch.nn as nn
import numpy as np
import torch


class SEQ2SEQ_Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 num_layers,
                 batch_first,
                 batch_size,
                 device,
                 encoder_type="LSTM"):
        super(SEQ2SEQ_Encoder, self).__init__()
        self.encoder = nn.LSTM(input_size,
                               embedding_size,
                               num_layers,
                               batch_first=batch_first,
                               device=device) if encoder_type == "LSTM" else nn.RNN(input_size,
                                                                                    embedding_size,
                                                                                    num_layers,
                                                                                    batch_first=batch_first,
                                                                                    device=device)
        self.device = device
        print("device of SEQ2SEQ_Encoder", self.device)
        self.encoder.h0 = torch.zeros(num_layers, batch_size, embedding_size).to(self.device)
        if encoder_type == "LSTM":
            self.encoder.c0 = torch.zeros(num_layers, batch_size, embedding_size).to(self.device)
        
    def forward(self, x, hidden):
        out, hidden = self.encoder(x, hidden)
        return out, hidden

class SEQ2SEQ_Decoder(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 output_size,
                 num_layers,
                 batch_first,
                 batch_size,
                 device,
                 decoder_type="LSTM",
                 attention=False,
                 causal=True):
        super(SEQ2SEQ_Decoder, self).__init__()
        self.decoder = nn.LSTM(input_size,
                               embedding_size,
                               num_layers,
                               batch_first=batch_first,
                               device=device) if decoder_type == "LSTM" else nn.RNN(input_size,
                                                                                    embedding_size,
                                                                                    num_layers,
                                                                                    batch_first=batch_first,
                                                                                    device=device)
        self.fc = nn.Linear(embedding_size, output_size).to(device)
        self.attention = attention
        self.causal = causal
        if attention:
            self.attention_layer = nn.MultiheadAttention(embedding_size, 4, batch_first=True, device=device).to(device)
        self.device = device
        print("device of SEQ2SEQ_decoder", self.device)
        self.decoder.h0 = torch.zeros(num_layers, batch_size, embedding_size).to(self.device)
        if decoder_type == "LSTM":
            self.decoder.c0 = torch.zeros(num_layers, batch_size, embedding_size).to(self.device)
        self.fc1 = nn.Linear(input_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, input_size)
        
    def forward(self, x, hidden, encoder_out):
        if self.attention:
            x = self.fc1(x)
            x, _ = self.attention_layer(x, encoder_out, encoder_out)
            x = self.fc2(x)
        # print("decoder forward 1", x.shape)
        out, hidden = self.decoder(x, hidden)
        out = self.fc(out)
        return out, hidden
        

class FK_SEQ2SEQ(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 num_layers,
                 batch_first,
                 batch_size,
                 output_size,
                 device,
                 encoder_type="LSTM",
                 decoder_type="LSTM",
                 domain_boundary=100,
                 pred_len=1,
                 attention=False,
                 teacher_forcing_ratio=0.75):
        super(FK_SEQ2SEQ, self).__init__()
        self.encoder = SEQ2SEQ_Encoder(input_size,
                                        embedding_size,
                                        num_layers,
                                        batch_first,
                                        batch_size,
                                        device,
                                        encoder_type)
        self.decoder = SEQ2SEQ_Decoder(input_size,
                                        embedding_size,
                                        output_size,
                                        num_layers,
                                        batch_first,
                                        batch_size,
                                        device,
                                        decoder_type,
                                        attention)
        self.device = device
        print("device of SEQ2SEQ", self.device)
        self.logstd = nn.Parameter(torch.tensor(np.ones(output_size)*0.1).to(self.device)).to(self.device)
        self.activation = nn.Tanh()
        self.pred_len = pred_len
        self.boundary = domain_boundary
        self.force_ratio = teacher_forcing_ratio
    
    # Allows for stateful or stateless LTSM
    def forward(self, x, gnd_truth, hidden, prob=False, mode='train'):
        # print("main forward", x.shape)
        encoder_out, encoder_hidden = self.encoder(x, hidden)
        decoder_input = x[:, -1:, :]
        outputs = [None for _ in range(self.pred_len)]
        decoder_hidden = encoder_hidden
        for i in range(self.pred_len):
            decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_out)
            if prob:
                distribution = torch.distributions.Normal(loc=decoder_out,
                                                        scale=torch.exp(self.logstd))
                decoder_out = distribution.rsample().float()
            decoder_out = torch.tanh(decoder_out) * self.boundary
            outputs[i] = decoder_out
            if np.random.random() < self.force_ratio and mode == 'train':
                decoder_input = gnd_truth[:, i:i+1, :]
            else:
                # add velocities
                decoder_input = decoder_out 
            # print("decoder input shape", decoder_input.shape, i)
        out = torch.cat(outputs, dim=1)
        return out, encoder_hidden