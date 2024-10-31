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

# input_size = 6
# hidden_size = 512
# num_layers = 3
# batch_size = 512
# output_size = 2

# input_size = 6
# hidden_size = 128
# num_layers = 1
# batch_size = 512
# output_size = 2

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
    # if 'mlp' in file_lower:
    #     continue
    # fine_tuning
    # if not (('b1024_e400_s25000_finetune_final_1730122927' in file_lower) or ('b1024_e400_s25000_len100_25000_blue_1730124674' in file_lower) or ('b1024_e400_s-1_len100_ema_0.8_1728875793' in file_lower)):
    #     continue
    # Purp 
    if (not (('LSTM_b1024_e400_s-1_len100_ema_0.8_1730123022' in file) 
             or ('RNN_b1024_e400_s-1_len100_ema_0.8_1730126891' in file) 
             or ('SEQ2SEQ_ATTENTION_b1024_e400_s-1_len100_ema_0.8_1728875793' in file)
             or ('SEQ2SEQ_b1024_e400_s-1_len100_ema_0.8_1728882683' in file))
        ):
        continue
    # if 'rnn' not in file_lower:
    #     continue
    # if 'ema0.2' in file_lower:
    #     ema = 0.2
    # elif 'ema0.5' in file_lower:
    #     ema = 0.5
    # elif 'ema0.8' in file_lower: 
    #     ema = 0.8
    # else:
    #     ema = 1
    ema = 0.8
    # if 'new' not in file_lower or 'time' not in file_lower:
    #     continue
    # if 'vel' not in file_lower and 'no_time' not in file_lower:
    #     continue
    # if 'raw' not in file_lower:
    #     continue
    input_size = 6
    # if 'vel' in file_lower:
    #     input_size = 8
    # if 'raw' in file_lower:
    #     input_size = 4
    
    print("input shape", input_size)
    # test_data_path = '../ml_data/test_data.csv'
    # if 'finetune' in file_lower:
    #     test_data_path = '../ml_data/purple_test_data.csv'
    # if 'cool' in file_lower or 'new' in file_lower:
    #     test_data_path = '../ml_data/purple_no_cool_down_test_data.csv'
    appendix = "saw_tooth_purp_vid"
    test_data_path = '../ml_data/'+ appendix + '.csv'
    print(test_data_path)
    filename = '../model_weights/new_weights/' + file
    print('File', file)
    outputs_df, test_df, r2_score, rmse = rollout(filename,
                                                  test_data_path,
                                                  input_size,
                                                  hidden_size,
                                                  num_layers,
                                                  batch_size,
                                                  output_size,
                                                  device,
                                                  ema=ema,
                                                  seq_len=100)
    fig = viz_graph(outputs_df, test_df, file, show_end=False)
    with open("../results/oct_31/"+file+"_"+appendix+".txt", 'w') as f:
        f.write("RMSE: " + str(rmse.item()) + '\n')
        f.write("R^2: " + str(r2_score.item()))
    outputs_df.to_csv("../results/oct_31/outputs/outputs_"+file+"_"+appendix+".csv")
    test_df.to_csv("../results/oct_31/test/test_"+file+"_"+appendix+".csv")
    
    fig.savefig("../results/oct_31/"+file+"_"+appendix+".jpg")
    fig.show()