import torch
import numpy as np

def load_data(data_dir='disparity'):
    """Load and format dataset .csv files containing stimuli, labels, and label values."""
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


def load_filters(data_dir='disparity'):
    """Load and format pretrained filters in .csv file."""

    filters = torch.as_tensor(
      np.loadtxt(data_dir / 'filters.csv', delimiter=',', dtype=np.float32)
    )

    return filters


