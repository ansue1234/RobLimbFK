import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from robo_limb_ml.models.fk_lstm import FK_LSTM
from robo_limb_ml.models.fk_mlp import FK_MLP
from robo_limb_ml.models.fk_rnn import FK_RNN
from robo_limb_ml.models.fk_seq2seq import FK_SEQ2SEQ
from robo_limb_ml.utils.utils import rollout, viz_graph
from tqdm import tqdm

input_size = 6
hidden_size = 512
num_layers = 3
batch_size = 512
output_size = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

files = os.listdir('../model_weights/new_weights')

for file in tqdm(files):
    file_lower = file.lower()
    print(file_lower)
    # skip RNN and MLP
    if 'rnn' in file_lower or 'mlp' in file_lower:
        continue
    # if 'vel' not in file_lower and 'no_time' not in file_lower:
    #     continue
    # if 'raw' not in file_lower:
    #     continue
    input_size = 6
    if 'vel' in file_lower:
        input_size = 8
    if 'raw' in file_lower:
        input_size = 4
    
    print("input shape", input_size)
    test_data_path = '../ml_data/test_data.csv'
    if 'finetune' in file_lower:
        test_data_path = '../ml_data/purple_test_data.csv'
    if 'cool' in file_lower:
        test_data_path = '../ml_data/purple_no_cool_down_test_data.csv'
        
    filename = '../model_weights/new_weights/' + file
    print('File', file)
    outputs_df, test_df, r2_score, rmse = rollout(filename,
                                                  test_data_path,
                                                  input_size,
                                                  hidden_size,
                                                  num_layers,
                                                  batch_size,
                                                  output_size,
                                                  device)
    fig = viz_graph(outputs_df, test_df, file, show_end=True)
    
    with open("../results/"+file+".txt", 'w') as f:
        f.write("RMSE: " + str(rmse.item()) + '\n')
        f.write("R^2: " + str(r2_score.item()))
    outputs_df.to_csv("../results/outputs_"+file+".csv")
    test_df.to_csv("../results/test_"+file+".csv")
    
    fig.savefig("../results/"+file+".jpg")
    fig.show()