import torch
import numpy as np

def load_data(data_dir='disparity'):

    stimuli = torch.as_tensor(
      np.loadtxt(data_dir / 'stimuli.csv', delimiter=',', dtype=np.float32)
    )

    labels = torch.as_tensor(
      np.loadtxt(data_dir / 'labels.csv', delimiter=',', dtype=int)
    )
    labels = labels - 1 # make 0-indexed

    values = torch.as_tensor(
      np.loadtxt(data_dir / 'label_values.csv', delimiter=',', dtype=np.float32)
    )

    return {'stimuli': stimuli, 'labels': labels, 'values': values}



