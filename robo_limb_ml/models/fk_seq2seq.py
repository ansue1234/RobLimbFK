import torch
import torch.nn as nn
import numpy as np


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
    
    def forward(self, hn, cn, x):
        out, (out_hn, out_cn) = self.encoder(x, (hn, cn))
        return out, out_hn, out_cn

class SEQ2SEQ_Decoder(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 output_size,
                 num_layers,
                 batch_first,
                 batch_size,
                 device,
                 decoder_type="LSTM"):
        super(SEQ2SEQ_Decoder, self).__init__()
        self.decoder = nn.LSTM(input_size,
                               embedding_size,
                               num_layers,
                               batch_first=batch_first,
                               device=device) if decoder_type == "LSTM" else nn.RNN(embedding_size,
                                                                                    embedding_size,
                                                                                    num_layers,
                                                                                    batch_first=batch_first,
                                                                                    device=device)
        self.fc = nn.Linear(embedding_size, output_size).to(device)
        self.device = device
        print("device of SEQ2SEQ_decoder", self.device)
        self.decoder.h0 = torch.zeros(num_layers, batch_size, embedding_size).to(self.device)
        if decoder_type == "LSTM":
            self.decoder.c0 = torch.zeros(num_layers, batch_size, embedding_size).to(self.device)
    
    def forward(self, hn, cn, x, encoder_out):
        out, (out_hn, out_cn) = self.decoder(x, (hn, cn))
        out = self.fc(out)
        return out, out_hn, out_cn
        

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
                 pred_len=10):
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
                                        decoder_type)
        self.device = device
        print("device of SEQ2SEQ", self.device)
        self.logstd = nn.Parameter(torch.tensor(np.ones(output_size)*0.1).to(self.device)).to(self.device)
        self.activation = nn.Tanh()
        self.pred_len = pred_len
        self.boundary = domain_boundary
    
    # Allows for stateful or stateless LTSM
    def forward(self, x, gnd_truth, hn, cn, prob=False, teacher_forcing_prob=0.5):
        encoder_out, encoder_hn, encoder_cn = self.encoder(hn, cn, x)
        decoder_input = x[:, -1:, :]
        outputs = [None for _ in range(self.pred_len)]
        decoder_hn, decoder_cn = encoder_hn, encoder_cn
        for i in range(self.pred_len):
            decoder_out, decoder_hn, decoder_cn = self.decoder(decoder_hn, decoder_cn, decoder_input, encoder_out)
            if prob:
                distribution = torch.distributions.Normal(loc=decoder_out,
                                                        scale=torch.exp(self.logstd))
                decoder_out = distribution.rsample().float()
            decoder_out = torch.tanh(decoder_out) * self.boundary
            outputs[i] = decoder_out
            if np.random.random() < teacher_forcing_prob:
                decoder_input = gnd_truth[:, i:i+1, :]
            else:
                decoder_input = decoder_out
        out = torch.cat(outputs, dim=1)
        return out, encoder_hn, encoder_cn