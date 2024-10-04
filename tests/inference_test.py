import pytest
import torch
import numpy as np
import amatorch.ama_class as cl
from amatorch.data import disparity_data, disparity_filters

@pytest.fixture(scope='module')
def data():
    return disparity_data()

@pytest.fixture(scope='module')
def filters():
    return disparity_filters()

@pytest.fixture(scope='module')
def responses_ref():
    return torch.as_tensor(
        np.loadtxt('./testing_data/dsp_responses.csv', delimiter=','),
        dtype=torch.float32
    )

@pytest.fixture(scope='module')
def log_likelihoods_ref():
    return torch.as_tensor(
        np.loadtxt('./testing_data/dsp_log_likelihoods.csv', delimiter=','),
        dtype=torch.float32
    )

@pytest.fixture(scope='module')
def posteriors_ref():
    return torch.tensor(
        np.loadtxt('./testing_data/dsp_posteriors.csv', delimiter=','),
        dtype=torch.float32
    )

@pytest.fixture(scope='module')
def estimates_ref():
    return torch.tensor(
        np.loadtxt('./testing_data/dsp_estimated_class.csv', delimiter=','),
        dtype=int
    )

def test_ama_gauss_responses(data, filters, responses_ref):
    """Test that AMA-Gauss can obtain responses to stimuli
    and that the responses are the correct shape."""
    n_filters = filters.shape[0]
    n_stimuli, n_channels, n_dim = data['stimuli'].shape

    ama = cl.AMAGauss(
        stimuli=data['stimuli'],
        labels=data['labels'],
        n_filters=2,
    )

    # Assign pretrained filters
    ama.filters = filters
    # Get responses to the stimuli
    responses = ama.responses(data['stimuli'])

    assert responses.shape == (n_stimuli, n_filters), 'Responses are not the correct shape'
    assert not torch.isnan(responses).any(), 'Responses are nan'
    assert torch.allclose(responses, responses_ref, atol=1e-6), 'Responses are not close to reference'

def test_ama_gauss_inference(data, filters, log_likelihoods_ref, posteriors_ref, estimates_ref):
    """Test that AMA-Gauss can perform inference on stimuli
    and that the outputs are close to the reference values."""
    n_stimuli, n_channels, n_dim = data['stimuli'].shape

    ama = cl.AMAGauss(
        stimuli=data['stimuli'],
        labels=data['labels'],
        n_filters=2,
    )

    # Assign pretrained filters
    ama.filters = filters

    # Get inference results
    with torch.no_grad():
        log_likelihoods = ama.log_likelihoods(data['stimuli'])
        posteriors = ama.posteriors(data['stimuli'])
        estimates = ama.estimates(data['stimuli'])

    assert not torch.isnan(log_likelihoods).any(), 'Log likelihoods are nan'
    assert not torch.isnan(posteriors).any(), 'Posteriors are nan'

    # Get relative norm of difference between reference and obtained values
    log_likelihoods_diff = torch.norm(log_likelihoods - log_likelihoods_ref, dim=-1) \
        / torch.norm(log_likelihoods_ref, dim=-1)
    posteriors_diff = torch.norm(posteriors - posteriors_ref, dim=-1) \
        / torch.norm(posteriors_ref, dim=-1)

    assert torch.allclose(log_likelihoods_diff, torch.zeros_like(log_likelihoods_diff), atol=1e-2), 'Log likelihoods are not close to reference'
    assert torch.allclose(posteriors_diff, torch.zeros_like(posteriors_diff), atol=1e-2), 'Posteriors are not close to reference'
    assert torch.sum(estimates != estimates_ref) < 15, 'Estimates are not close to reference'
