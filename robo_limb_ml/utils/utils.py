import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from robo_limb_ml.models.fk_lstm import FK_LSTM
from robo_limb_ml.models.fk_mlp import FK_MLP
from robo_limb_ml.models.fk_rnn import FK_RNN
from robo_limb_ml.models.fk_seq2seq import FK_SEQ2SEQ
from tqdm import tqdm
from torcheval.metrics import R2Score

def get_velocities(data):
    """
    data: pd.DataFrame
    calculates the velocities from the thetas
    convert milliseconds to seconds
    returns a pd.DataFrame with the velocities
    """
    data['time_begin'] = (data['time'] - data['time'][0])/1000
    # one trajectory is defined as pts from cool down to cool down
    data['dt'] = data['time'].diff()
    data['new_traj'] = np.abs(data['dt']) > 10000
    data['traj_num'] = data['new_traj'].cumsum().ffill().astype(int)
    dfs = []
    for _, group in data.groupby('traj_num'):
        group['delta_theta_x'] = group['theta_x'].diff()
        group['delta_theta_y'] = group['theta_y'].diff()
        group['time_begin_traj'] = (group['time'] - group['time'].iloc[0])/1000
        group['delta_t'] = group['time'].diff()/1000
        group['vel_x'] = group['delta_theta_x'] / group['delta_t']
        group['vel_y'] = group['delta_theta_y'] / group['delta_t']
        # drop first 5 samples
        group = group.iloc[5:]
        group = group.dropna()
        dfs.append(group)
    data_vel = (pd.concat(dfs, ignore_index=True)
                  .drop(columns=['new_traj',
                                 'dt',
                                 'traj_num', 
                                 'delta_theta_x', 
                                 'delta_theta_y']))
    return data_vel

def concat_data(directory, vel=False):
    """
    dir: str
    concatenates the data from the csv files in the directory
    returns a pd.DataFrame
    """
    files = os.listdir(directory)
    dfs = []
    for file in files:
        data = pd.read_csv(os.path.join(directory, file))
        if vel:
            data = get_velocities(data)
        dfs.append(data)
    data = pd.concat(dfs, ignore_index=True)
    return data

def rollout(model_path,
            test_data_path,
            input_size,
            hidden_size,
            num_layers,
            batch_size,
            output_size,
            device):
    model_path_lower = model_path.lower()
    model_type = 'SEQ2SEQ' if 'seq2' in model_path_lower else 'LSTM'
    attention = True if 'attention' in model_path_lower else False
    stateful = False if 'stateless' in model_path_lower else False
    if 'len10' in model_path_lower:
        seq_len = 10
    elif 'new' in model_path_lower:
        seq_len = 100
    else:
        seq_len = 50
    # seq_len = 10 if 'len10' in model_path_lower else 50
    vel = True if 'vel' in model_path_lower or 'no_time' in model_path_lower else False
    no_time = True if 'no_time' in model_path_lower else False
    # print("model_path", model_path_lower)
    if 'raw' in model_path_lower:
        vel = False
        no_time = True
    vel = True
    no_time = True
    if model_type == 'LSTM':
        model = FK_LSTM(input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_size=batch_size,
                output_size=output_size,
                device=device,
                batch_first=True).to(device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.h0 = torch.zeros(num_layers, 1, hidden_size).to(device=device)
        model.c0 = torch.zeros(num_layers, 1, hidden_size).to(device=device)
    elif model_type == 'SEQ2SEQ':
        model = FK_SEQ2SEQ(input_size=input_size,
                   embedding_size=hidden_size,
                   num_layers=num_layers,
                   batch_size=batch_size,
                   output_size=output_size,
                   device=device,
                   batch_first=True,
                   encoder_type='LSTM',
                   decoder_type='LSTM',
                   attention=attention,
                   pred_len=1,
                   teacher_forcing_ratio=0.0).to(device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    if not vel:
        test_df = pd.read_csv(test_data_path).dropna()
        col_1 = test_df.pop('vel_x')
        test_df.insert(7, col_1.name, col_1)
        col_2 = test_df.pop('vel_y')
        test_df.insert(7, col_2.name, col_2)
    else:
        test_df = pd.read_csv(test_data_path).dropna()
    test_tensor = torch.tensor(test_df.values.copy(), dtype=torch.float32).to(device=device)
    # obs_tensor = torch.tensor(test_df.drop(columns=["vel_x", "vel_y"]).values, dtype=torch.float32).to(device=device)
    outputs = torch.zeros(test_df.shape).to(device=device)
    outputs[:seq_len] = test_tensor[:seq_len].clone()
    hn = torch.zeros(num_layers, 1, hidden_size).to(device=device)
    cn = torch.zeros(num_layers, 1, hidden_size).to(device=device)
    hidden = (hn, cn)
    with torch.no_grad():
        for i in tqdm(range(seq_len, test_df.shape[0])):
            # print("vel", vel)
            # print("no time", no_time)
            if vel and not no_time:
                data = outputs[i - seq_len:i]
            elif not vel and no_time:
                data = outputs[i - seq_len:i, 2:-2]
            elif not vel and not no_time:
                data = outputs[i - seq_len:i, :-2]
            elif vel and no_time:
                data = outputs[i - seq_len:i, 2:]
            # print("data shape", data.shape)
            if not vel:    
                time_begin, time_begin_traj, theta_x, theta_y, X_throttle, Y_throttle, vel_x, vel_y  = outputs[i - 1]
            else:
                time_begin, time_begin_traj, theta_x, theta_y, vel_x, vel_y, X_throttle, Y_throttle  = outputs[i - 1]
                
            if stateful:
                if model_type == 'LSTM':
                    delta_states, hn, cn = model(data.unsqueeze(0), hn, cn)
                elif model_type == 'SEQ2SEQ':
                    delta_states, hidden = model(data.unsqueeze(0), None, hidden, mode='test')
            else:
                if model_type == 'LSTM':
                    delta_states, _, _ = model(data.unsqueeze(0), hn, cn)
                else:
                    delta_states, _ = model(data.unsqueeze(0), None, hidden, mode='test')
            delta_states = delta_states.squeeze()
            
            pred_theta_x, pred_theta_y, pred_vel_x, pred_vel_y = delta_states[0] + theta_x, delta_states[1] + theta_y, delta_states[2] + vel_x, delta_states[3] + vel_y
            if not vel:
                time_begin_1, time_begin_traj_1, _, _, X_throttle_1, Y_throttle_1, _, _ = test_tensor[i]
                outputs[i] = torch.tensor([time_begin_1, time_begin_traj_1, pred_theta_x, pred_theta_y, X_throttle_1, Y_throttle_1, pred_vel_x, pred_vel_y]).to(device=device)
            else:
                time_begin_1, time_begin_traj_1, _, _, _, _, X_throttle_1, Y_throttle_1 = test_tensor[i]
                outputs[i] = torch.tensor([time_begin_1, time_begin_traj_1, pred_theta_x, pred_theta_y, pred_vel_x, pred_vel_y, X_throttle_1, Y_throttle_1]).to(device=device)
            
    outputs_df = pd.DataFrame(outputs.cpu().detach().numpy(), columns=test_df.columns)
    output_states = torch.tensor(outputs_df[['theta_x', 'theta_y', 'vel_x', 'vel_y']].values, dtype=torch.float32).to(device=device)
    test_states = torch.tensor(test_df[['theta_x', 'theta_y', 'vel_x', 'vel_y']].values, dtype=torch.float32).to(device=device)
    
    metric = R2Score()
    metric.update(test_states, output_states)
    r2_score = metric.compute()
    print("R^2", r2_score.item())
    rmse = torch.sqrt(nn.MSELoss()(test_states, output_states))
    print("RMSE", rmse.item())
    
    return outputs_df, test_df, r2_score, rmse

def viz_graph(outputs_df, test_df, run_name, display_window=1500, show_end=False):
    if show_end:
        begin = test_df.shape[0] - display_window
        end = test_df.shape[0]
    else:
        begin = 0
        end = display_window
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].plot(test_df["time_begin"][begin:end], test_df["theta_x"][begin:end], label="Actual")
    axs[0, 0].plot(outputs_df["time_begin"][begin:end], outputs_df["theta_x"][begin:end], label="Predicted")
    axs[0, 0].set_title("Theta X")
    axs[0, 0].legend()
    axs[0, 1].plot(test_df["time_begin"][begin:end], test_df["theta_y"][begin:end], label="Actual")
    axs[0, 1].plot(outputs_df["time_begin"][begin:end], outputs_df["theta_y"][begin:end], label="Predicted")
    axs[0, 1].set_title("Theta Y")
    axs[0, 1].legend()
    axs[1, 0].plot(test_df["time_begin"][begin:end], test_df["vel_x"][begin:end], label="Actual")
    axs[1, 0].plot(outputs_df["time_begin"][begin:end], outputs_df["vel_x"][begin:end], label="Predicted")
    axs[1, 0].set_title("Vel X")
    axs[1, 0].legend()
    axs[1, 1].plot(test_df["time_begin"][begin:end], test_df["vel_y"][begin:end], label="Actual")
    axs[1, 1].plot(outputs_df["time_begin"][begin:end], outputs_df["vel_y"][begin:end], label="Predicted")
    axs[1, 1].legend()
    axs[1, 1].set_title("Vel Y")
    fig.suptitle(run_name, fontsize=16)
    return fig