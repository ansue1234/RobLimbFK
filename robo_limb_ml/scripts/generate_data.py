import torch
import numpy as np
import pandas as pd
import sys
from robo_limb_ml.models.fk_model import PsuedoFKModel
from tqdm import tqdm

"""
To generate synthetic data to go through the pipeline
"""

if __name__ == "__main__":
    sys.path.append('..')
    # Generate some data
    # input: 4 values, pwm percentage for 4 SMAs (0-100%) or 0-1
    # output: 2 values, 2 angles (degrees)
    n_samples = 2**20
    # 4 Current SMA PWM values, 4 Previous SMA PWM values, 2 previous angles, and time since last command 
    input_size = 11

    hidden_sizes = [64, 128, 128, 64]
    # 2 angles
    output_size = 2

    distribution = 't'
    distribution_params = {'df': 2, 'std': 0.1}
    model = PsuedoFKModel(input_size=input_size,
                           hidden_sizes=hidden_sizes,
                           output_size=output_size,
                           distribution_name=distribution,
                           distribution_params=distribution_params)
    init_pwm = torch.rand(8)
    init_angles = torch.rand(2)*np.pi/2 - np.pi/4
    init_t = torch.randint(100, 1000, (1,))
    init_inputs = torch.cat((init_pwm, init_angles, init_t), 0)

    results = torch.cat((init_inputs, init_angles), 0)
    final_results = None
    for i in tqdm(range(1, n_samples + 1)):
        output = model.forward_prob(init_inputs)
        result = torch.cat((init_inputs, output), 0)
        results = torch.vstack((results, result))

        next_pwm = torch.rand(4)
        prev_pwm = init_inputs[0:4]
        next_t = torch.randint(100, 1000, (1,))
        init_inputs = torch.cat((next_pwm, init_inputs[0:4], output, next_t), 0)
        
        if i % 2**14 == 0:
            if final_results is not None:
                final_results = torch.vstack((final_results, results))
            else:
                final_results = results
            df = pd.DataFrame(final_results.cpu().detach().numpy(), columns=['sma1',
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
                                                                       'angle2']).drop_duplicates()
            df.to_csv('../data/data.csv', index=False)
            results = torch.cat((init_inputs, output), 0)
    print('Data saved to data.csv')

