import pytest
import torch
from amatorch.data import disparity_data

def test_disparity_data_loading():
    # Load the data using your function
    data = disparity_data()

    # Check that the data is loaded correctly
    assert 'stimuli' in data
    assert 'labels' in data
    assert 'values' in data

    # Check that the data is of the correct type (torch.Tensor)
    assert isinstance(data['stimuli'], torch.Tensor)
    assert isinstance(data['labels'], torch.Tensor)
    assert isinstance(data['values'], torch.Tensor)

