##################

# TESTS THAT THE AMA CLASS INITIALIZES AND TRAINS
# FOR THE RELEASE BEFORE REFACTORING
#
##################

import pytest
import torch
import amatorch.ama_class as cl
import amatorch.optim as optim
from amatorch.data import disparity_data
import amatorch.utilities as au

def test_inference():

    # Load the data
    data_dict = disparity_data()
    stimuli = data_dict['stimuli']
    labels = data_dict['labels']
    values = data_dict['values']
    n_classes = len(values)
    n_channels = stimuli.shape[-2]
    n_dim = stimuli.shape[-1]

    ama = cl.AMAGauss(
      stimuli=stimuli,
      labels=labels,
      n_filters=2,
      values=values,
    )

    # Assert that the filters are initialized with the correct shape
    assert ama.filters.shape == (2, n_channels, n_dim), 'Filters are not the correct shape'
    # Assert that stimulus statistics are initialized with the correct shape
    assert ama.stimulus_statistics.means.shape == (n_classes, n_channels * n_dim), 'Means are not the correct shape'
    assert ama.stimulus_statistics.covariances.shape == (n_classes, n_channels * n_dim, n_channels * n_dim), 'Covariances are not the correct shape'
    # Assert that response statistics are initialized with the correct shape
    assert ama.response_statistics.means.shape == (n_classes, 2), 'Means are not the correct shape'
    assert ama.response_statistics.covariances.shape == (n_classes, 2, 2), 'Covariances are not the correct shape'

    # Get responses to the stimuli
    responses = ama.responses(stimuli)

    # Assert that the responses are the correct shape
    assert responses.shape == (len(stimuli), 2), 'Responses are not the correct shape'
    # Responses are not nan
    assert not torch.isnan(responses).any(), 'Responses are nan'


