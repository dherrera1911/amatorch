import pytest
import torch
import amatorch.ama_class as cl
from amatorch.data import disparity_data, disparity_filters

def test_ama_gauss_responses():
    """Test that AMA-Gauss can obtain responses to stimuli
    and that the responses are the correct shape."""
    data = disparity_data()
    filters = disparity_filters()

    n_filters = filters.shape[0]
    n_stimuli, n_channels, n_dim = data['stimuli'].shape

    ama = cl.AMAGauss(
      stimuli=data['stimuli'],
      labels=data['labels'],
      n_filters=2,
    )

    # Assert that the filters are initialized with the correct shape
    assert ama.filters.shape == (2, n_channels, n_dim), 'Filters are not the correct shape'

    # Assign pretrained filters
    ama.filters = filters
    # Get responses to the stimuli
    responses = ama.responses(data['stimuli'])

    # Assert that the responses are the correct shape
    assert responses.shape == (n_stimuli, n_filters), 'Responses are not the correct shape'
    assert not torch.isnan(responses).any(), 'Responses are nan'
    assert not (responses.norm(dim=-1) > n_filters).any(), 'Responses are too large'


def test_ama_gauss_inference():
    """Test that AMA-Gauss can obtain responses to stimuli
    and that the responses are the correct shape."""
    data = disparity_data()
    filters = disparity_filters()

    n_filters = filters.shape[0]
    n_stimuli, n_channels, n_dim = data['stimuli'].shape

    ama = cl.AMAGauss(
      stimuli=data['stimuli'],
      labels=data['labels'],
      n_filters=2,
    )

    # Assert that the filters are initialized with the correct shape
    assert ama.filters.shape == (2, n_channels, n_dim), 'Filters are not the correct shape'

    # Assign pretrained filters
    ama.filters = filters
    # Get responses to the stimuli
    responses = ama.responses(data['stimuli'])

    # Assert that the responses are the correct shape
    assert responses.shape == (n_stimuli, n_filters), 'Responses are not the correct shape'
    assert not torch.isnan(responses).any(), 'Responses are nan'
    assert not (responses.norm(dim=-1) > n_filters).any(), 'Responses are too large'

