import os

import numpy as np
import pytest
import torch

from amatorch.data import disparity_data, disparity_filters
from amatorch.models import AMAGauss

@pytest.fixture(scope="module")
def data():
    return disparity_data()

@pytest.fixture(scope="module")
def filters():
    return disparity_filters()


def test_to_dtype(data, filters):
    """Test that AMA-Gauss can obtain responses to stimuli
    and that the responses are the correct shape."""
    ama = AMAGauss(
        stimuli=data["stimuli"],
        labels=data["labels"],
        n_filters=2,
    )

    # Get responses to the stimuli
    responses = ama.responses(data["stimuli"])

    assert responses.dtype == torch.float32, "Posteriors are not float32"

    # Move to float64
    data['stimuli'] = data['stimuli'].double()
    ama.to(data['stimuli'])

    responses = ama.responses(data["stimuli"])
    posteriors = ama.posteriors(data["stimuli"])

    assert posteriors.dtype == torch.float64, "Posteriors are not float64"
    assert responses.dtype == torch.float64, "Responses are not float64"
