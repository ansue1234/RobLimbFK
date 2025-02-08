from robo_limb_ml.models.fk_lstm import FK_LSTM
import torch.nn as nn
import torch

lstm = FK_LSTM(input_size=3, hidden_size=512, num_layers=2, batch_first=True, batch_size=1, output_size=4, device='cpu')
new_lstm = nn.Sequential(*(list(lstm.children())[:-2]))
fake_data = torch.ones(1, 2, 10, 3)
hn = torch.zeros(2, 1, 512)
cn = torch.zeros(2, 1, 512)
nn_lstm = nn.LSTM(3, 512, 2, batch_first=True)
print(nn_lstm(fake_data))
