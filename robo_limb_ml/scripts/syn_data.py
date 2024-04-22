import torch
import numpy as np
import pandas as pd
import sys
from robo_limb_ml.utils.data_loader import DataLoader
from tqdm import tqdm


if __name__ == "__main__":
    data_loader = DataLoader(file_path='../data/data.csv',
                             batch_size=2048,
                             num_samples=2**14,
                             device='cpu')
    data, labels = data_loader.get_all_data()
    data = data.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    data_original = data.copy()
    labels_original = labels.copy()

    # print(data.shape)
    # print(labels.shape)

    for i in range(8):
        data_to_cp = data_original[:, :-3]
        data_time = data_original[:, -1:]
        initial_data_label = data_original[0, 8:10]
        generated_labels = np.random.normal(loc=labels_original, scale=0.1)
        generated_labels = np.vstack((np.random.normal(loc=initial_data_label, scale=0.1),
                                      generated_labels))
        generated_data = np.hstack((data_to_cp, generated_labels[:-1], data_time))
        labels = np.vstack((labels, generated_labels[1:]))
        data = np.vstack((data, generated_data))
        print(data.shape)
    
    all_data = np.hstack((data, labels))
    df = pd.DataFrame(all_data, columns=['sma1',
                                         'sma2',
                                         'sma3',
                                         'sma4',
                                         'sma1_p',
                                         'sma2_p',
                                         'sma3_p',
                                         'sma4_p',
                                         'angle1_p',
                                         'angle2_p',
                                         't',
                                         'angle1',
                                         'angle2'])
    # print(df.shape)
    df.to_csv('../data/syn_data.csv', index=False)




