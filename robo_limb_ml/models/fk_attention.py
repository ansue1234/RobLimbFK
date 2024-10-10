import torch
import torch.nn as nn
import numpy as np

class FK_GPT(nn.Module):
    def __init__(self, input_dim, embed_dim, out_dim, num_heads, num_layers, max_seq_length, device, sin_pos=False):
        super(FK_GPT, self).__init__()
        self.embed = nn.Linear(input_dim, embed_dim).to(device)
        self.device = device
        if not sin_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim).to(device)).to(device)
        else:
            self.pos_embed = self.create_positional_encoding(max_seq_length, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads).to(device)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers).to(device)

        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, out_dim).to(device))  # Predicting Δstate

    def forward(self, input_seq):
        embedding = self.embed(input_seq)
        x = embedding + self.pos_embed[:, :input_seq.size(1), :]
        x = x.permute(1, 0, 2)  # Transformer expects sequence first
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(self.device)
        output = self.transformer_decoder(x, x, tgt_mask=tgt_mask)
        output = output.permute(1, 0, 2)
        last_output = output[:, -1:, :]
        pred = self.output_layer(last_output)
        return pred

    def create_positional_encoding(self, max_seq_length, embed_dim):
        position = torch.arange(0, max_seq_length).unsqueeze(1)  # Shape: (max_seq_length, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_length, embed_dim).to(self.device)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_length, embed_dim)
        return pe  # This is a fixed buffer and does not require gradients