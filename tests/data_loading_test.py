import torch

from amatorch.datasets import disparity_data, disparity_filters


def test_disparity_data_loading():
    """Test that disparity data is loaded correctly."""
    data = disparity_data()

    assert "stimuli" in data
    assert "labels" in data
    assert "values" in data

    assert isinstance(data["stimuli"], torch.Tensor)
    assert isinstance(data["labels"], torch.Tensor)
    assert isinstance(data["values"], torch.Tensor)
    assert data["stimuli"].dim() == 3


def test_disparity_filter_loading():
    """Test that disparity filters are loaded correctly."""
    filters = disparity_filters()

    assert isinstance(filters, torch.Tensor)
    assert filters.dim() == 3
