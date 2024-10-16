import os

import numpy as np
import pytest
import torch

from amatorch.datasets import disparity_data, disparity_filters
from amatorch.models import AMAGauss

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testing_data")


def ref_data_file(file_name):
    return os.path.join(TEST_DATA_DIR, file_name)


@pytest.fixture(scope="module")
def data():
    return disparity_data()


@pytest.fixture(scope="module")
def filters():
    return disparity_filters()


@pytest.fixture(scope="module")
def responses_ref():
    return torch.as_tensor(
        np.loadtxt(ref_data_file("dsp_responses.csv"), delimiter=","),
        dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def log_likelihoods_ref():
    return torch.as_tensor(
        np.loadtxt(ref_data_file("dsp_log_likelihoods.csv"), delimiter=","),
        dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def posteriors_ref():
    return torch.tensor(
        np.loadtxt(ref_data_file("dsp_posteriors.csv"), delimiter=","),
        dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def estimates_ref():
    return torch.tensor(
        np.loadtxt(ref_data_file("dsp_estimated_class.csv"), delimiter=","), dtype=int
    )


class TestAMAGaussResponses:
    def test_response_values(self, data, filters, responses_ref):
        """Test that AMA-Gauss can obtain responses to stimuli
        and that the responses are the correct shape."""
        n_filters = filters.shape[0]
        n_stimuli, n_channels, n_dim = data["stimuli"].shape

        ama = AMAGauss(
            stimuli=data["stimuli"],
            labels=data["labels"],
        )

        # Assign pretrained filters
        ama.filters = filters
        # Get responses to the stimuli
        with torch.no_grad():
            responses = ama.responses(data["stimuli"])

        assert responses.shape == (
            n_stimuli,
            n_filters,
        ), "Responses are not the correct shape"
        assert not torch.isnan(responses).any(), "Responses are nan"
        assert torch.allclose(
            responses, responses_ref, atol=1e-6
        ), "Responses are not close to reference"

    def test_response_shapes(self, data, filters, responses_ref):
        """Test that responses can be obtained for stimuli of
        different shapes preceding channels and pixels."""
        n_filters = filters.shape[0]
        n_stimuli, n_channels, n_dim = data["stimuli"].shape

        ama = AMAGauss(
            stimuli=data["stimuli"],
            labels=data["labels"],
        )

        # Assign pretrained filters
        ama.filters = filters

        # Get responses to single stimulus
        stimulus_single = data["stimuli"][0]
        with torch.no_grad():
            responses_single = ama.responses(stimulus_single)

        assert responses_single.shape[0] == (
            n_filters
        ), "Responses to single stimulus are not the correct shape"

        # Get responses to stimuli with different preceding dimensions
        stimuli_2d = data["stimuli"][:100].reshape(2, 50, n_channels, n_dim)
        with torch.no_grad():
            responses_2d = ama.responses(stimuli_2d)

        assert responses_2d.shape == (
            2,
            50,
            n_filters,
        ), "Responses to grouped stimuli are not the correct shape"
        assert torch.allclose(
            responses_2d[0, 0], responses_single
        ), "Responses to grouped stimuli are not the same as to single stimulus"


class TestAMAGaussInference:
    def test_inference_values(
        self, data, filters, log_likelihoods_ref, posteriors_ref, estimates_ref
    ):
        """Test that AMA-Gauss can perform inference on stimuli
        and that the outputs are close to the reference values."""
        n_stimuli, n_channels, n_dim = data["stimuli"].shape

        ama = AMAGauss(
            stimuli=data["stimuli"],
            labels=data["labels"],
            n_filters=2,
        )

        # Assign pretrained filters
        ama.filters = filters

        # Get inference results
        with torch.no_grad():
            log_likelihoods = ama.log_likelihoods(data["stimuli"])
            posteriors = ama.posteriors(data["stimuli"])
            estimates = ama.estimates(data["stimuli"])

        assert not torch.isnan(log_likelihoods).any(), "Log likelihoods are nan"
        assert not torch.isnan(posteriors).any(), "Posteriors are nan"

        # Get relative norm of difference between reference and obtained values
        log_likelihoods_diff = torch.norm(
            log_likelihoods - log_likelihoods_ref, dim=-1
        ) / torch.norm(log_likelihoods_ref, dim=-1)
        posteriors_diff = torch.norm(posteriors - posteriors_ref, dim=-1) / torch.norm(
            posteriors_ref, dim=-1
        )

        assert torch.allclose(
            log_likelihoods_diff, torch.zeros_like(log_likelihoods_diff), atol=1e-2
        ), "Log likelihoods are not close to reference"
        assert torch.allclose(
            posteriors_diff, torch.zeros_like(posteriors_diff), atol=1e-2
        ), "Posteriors are not close to reference"
        assert (
            torch.sum(estimates != estimates_ref) < 15
        ), "Estimates are not close to reference"

    def test_inference_shapes(
        self, data, filters, log_likelihoods_ref, posteriors_ref, estimates_ref
    ):
        """Test that AMA-Gauss can perform inference on stimuli
        and that the outputs are close to the reference values."""
        n_stimuli, n_channels, n_dim = data["stimuli"].shape
        n_classes = data["values"].shape[0]

        ama = AMAGauss(
            stimuli=data["stimuli"],
            labels=data["labels"],
            n_filters=2,
        )

        # Assign pretrained filters
        ama.filters = filters[:2]

        # Get responses to single stimulus
        stimulus_single = data["stimuli"][0]
        with torch.no_grad():
            log_likelihoods_single = ama.log_likelihoods(stimulus_single)
            posteriors_single = ama.posteriors(stimulus_single)
            estimates_single = ama.estimates(stimulus_single)

        assert (
            log_likelihoods_single.shape[0] == n_classes
        ), "Log likelihoods to single stimulus are not the correct shape"
        assert (
            posteriors_single.shape[0] == n_classes
        ), "Posteriors to single stimulus are not the correct shape"
        assert (
            estimates_single.shape == ()
        ), "Estimates to single stimulus are not the correct shape"

        # Get responses to stimuli with different preceding dimensions
        stimuli_2d = data["stimuli"][:100].reshape(2, 50, n_channels, n_dim)
        with torch.no_grad():
            log_likelihoods_2d = ama.log_likelihoods(stimuli_2d)
            posteriors_2d = ama.posteriors(stimuli_2d)
            estimates_2d = ama.estimates(stimuli_2d)

        assert log_likelihoods_2d.shape == (
            2,
            50,
            n_classes,
        ), "Log likelihoods to grouped stimuli are not the correct shape"
        assert posteriors_2d.shape == (
            2,
            50,
            n_classes,
        ), "Posteriors to grouped stimuli are not the correct shape"
        assert estimates_2d.shape == (
            2,
            50,
        ), "Estimates to grouped stimuli are not the correct shape"
