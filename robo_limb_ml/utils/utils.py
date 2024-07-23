import numpy as np
import pandas as pd
import os

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