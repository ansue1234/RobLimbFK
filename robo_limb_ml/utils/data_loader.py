import torch
import numpy as np
import pandas as pd

class DataLoader():
    """
    class to prepared and process time series data fitting for LSTMs, RNNs
    """
    # pass in the input features and output features here
    def __init__(self,
                 file_path,
                 batch_size,
                 device,
                 predict_len=1,
                 seq_len=50,
                 num_samples=-1,
                 input_features=['time_begin',
                                 'time_begin_traj',
                                 'theta_x',
                                 'theta_y',
                                 'X_throttle',
                                 'Y_throttle'],
                 output_features=['delta_theta_x',
                                  'delta_theta_y',
                                  'delta_vel_x',
                                  'delta_vel_y'],
                 pad=False):
        """seq_len must be greater than or equal to predict_len"""
        assert seq_len >= predict_len
        self.data = pd.read_csv(file_path).iloc[:num_samples]
        self.batch_size = batch_size
        self.device = device
        self.n_samples = self.data.shape[0]
        # self.n_batches = self.n_samples // self.batch_size
        self.pred_len = predict_len
        self.seq_len = seq_len
        self.input_features = input_features
        self.output_features = output_features
        self.current_batch = 0
        self.pad = pad
        self.input_dim = len(input_features)
        self.output_dim = len(output_features)

        self._calc_label()
        self._format_data()

    def _calc_label(self):
        """
        calculates the label aka the difference between the current and next state
        each row in the data currently includes:
        obs_t, action_t
        we want to predict the difference between obs_t and obs_t+1 relative to each trajectory

        """
        data = self.data
        # do grouping by trajectory
        data['dt'] = data['time_begin'].diff()
        data['new_traj'] = np.abs(data['dt']) > 10
        data['new_set'] = data['dt'] < 0
        data['traj_num'] = data['new_traj'].cumsum().ffill().astype(int)
        data['set_num'] = data['new_set'].cumsum().ffill().astype(int)
        dfs = []
        for _, group in data.groupby('traj_num'):
            group['delta_theta_x'] = group['theta_x'].diff().shift(-1)
            group['delta_theta_y'] = group['theta_y'].diff().shift(-1)
            group['delta_vel_x'] = group['vel_x'].diff().shift(-1)
            group['delta_vel_y'] = group['vel_y'].diff().shift(-1)
            group = group.dropna()
            dfs.append(group)
        # pad to multiples of batch size for each set, 
        # so we can reset states after each set. Each set is defines as data within the same file
        # i.e. when time_begin resets. extra data is padded with zeros
        data_diff = (pd.concat(dfs, ignore_index=True)
                       .drop(columns=['new_traj',
                                      'dt',
                                      'traj_num',
                                      'new_set']))
        if self.pad:
            dfs = []
            for _, group in data_diff.groupby('set_num'):
                if group.shape[0] % self.batch_size != 0:
                    pad_len = self.batch_size - group.shape[0] % self.batch_size
                    pad = pd.DataFrame(np.zeros((pad_len, group.shape[1])), columns=group.columns)
                    group = pd.concat([group, pad])
                dfs.append(group)
            data_diff = pd.concat(dfs, ignore_index=True)
        data_diff = data_diff.dropna()
        self.data = data_diff
        return data_diff

    def _format_data(self):
        """
        formats the data into sequences of length seq_len
        into shape (n_samples, seq_len, n_features)
        """
        data = self.data
        # print(data.columns)
        input_data = data[self.input_features]
        output_data = data[self.output_features]
        input_data = input_data.values
        output_data = output_data.values
        num_sequences = input_data.shape[0] - self.seq_len - self.pred_len
        self.input_data = np.array([input_data[i:i+self.seq_len] for i in range(num_sequences)])
        output_data = output_data[self.seq_len-1:]
        throttle_data = input_data[self.seq_len-1:, -2:]
        self.output_data = np.array([output_data[i:i+self.pred_len] for i in range(num_sequences)])
        self.throttle_data = np.array([throttle_data[i:i+self.pred_len] for i in range(num_sequences)])
        self.n_batches = self.input_data.shape[0] // self.batch_size

    def get_batch(self):
        if self.current_batch == self.n_batches:
            self.current_batch = 0
        # if (self.current_batch+1)*self.batch_size > self.n_samples:
        #     data = self.input_data[self.current_batch*self.batch_size:]
        #     labels = self.output_data[self.current_batch*self.batch_size:]
        # else:

        data = torch.tensor(self.input_data[self.current_batch*self.batch_size:(self.current_batch+1)*self.batch_size]).to(device=self.device).float()
        labels = torch.tensor(self.output_data[self.current_batch*self.batch_size:(self.current_batch+1)*self.batch_size]).to(device=self.device).float()
        set_num = self.data['set_num'].values[self.current_batch*self.batch_size]
        # print(self.input_data.shape, self.output_data.shape)
        self.current_batch += 1
        return data, labels, set_num
    
    def get_batch_rollout(self):
        if self.current_batch == self.n_batches:
            self.current_batch = 0
        data = torch.tensor(self.input_data[self.current_batch*self.batch_size:(self.current_batch+1)*self.batch_size]).to(device=self.device).float()
        labels = torch.tensor(self.output_data[self.current_batch*self.batch_size:(self.current_batch+1)*self.batch_size]).to(device=self.device).float()
        throttle = torch.tensor(self.throttle_data[self.current_batch*self.batch_size:(self.current_batch+1)*self.batch_size]).to(device=self.device).float()
        set_num = self.data['set_num'].values[self.current_batch*self.batch_size]
        # print(self.input_data.shape, self.output_data.shape)
        self.current_batch += 1
        return data, labels, throttle, set_num
        
    def get_n_batches(self):
        return self.n_batches
 
    