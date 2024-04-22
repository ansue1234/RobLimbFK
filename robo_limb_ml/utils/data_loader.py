import torch
import numpy as np
import pandas as pd

class DataLoader():
    def __init__(self, file_path, batch_size, device, num_samples=-1):
        self.data = pd.read_csv(file_path).iloc[:num_samples]
        self.batch_size = batch_size
        self.device = device
        self.data = np.array(self.data.values, dtype=np.float32)
        self.n_samples = self.data.shape[0]
        self.n_batches = self.n_samples // self.batch_size
        self.current_batch = 0

    def get_batch(self):
        if self.current_batch == self.n_batches:
            self.current_batch = 0
        start = self.current_batch * self.batch_size
        end = (self.current_batch + 1) * self.batch_size
        self.current_batch += 1
        # training_data = torch.from_numpy(np.expand_dims(self.data[start:end, :-2], axis=1)).to(device=self.device)
        training_data = torch.from_numpy(self.data[start:end, :-2]).to(device=self.device)
        training_labels = torch.from_numpy(self.data[start:end, -2:]).to(device=self.device)
        return training_data, training_labels 

    def get_all_data(self):
        # training_data = torch.from_numpy(np.expand_dims(self.data[:, :-2], axis=1)).to(device=self.device)
        training_data = torch.from_numpy(self.data[:, :-2]).to(device=self.device)
        training_labels = torch.from_numpy(self.data[:, -2:]).to(device=self.device)
        return training_data, training_labels 
    
    def get_n_samples(self):
        return self.n_samples

    def get_n_batches(self):
        return self.n_batches