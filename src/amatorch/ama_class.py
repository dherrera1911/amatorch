from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tfun
from torch.nn.utils.parametrize import register_parametrization
from torch.distributions.multivariate_normal import MultivariateNormal
from amatorch import normalization
from amatorch import inference
from amatorch import utilities as au

#####################
# PARENT AMA CLASS
#####################

class AMA(ABC, nn.Module):
    def __init__(self, n_dim, n_filters, priors, n_channels=1):
        """ Abstract AMA parent class.
        -----------------
        Arguments:
        -----------------
          - n_dim: Number of dimensions of inputs
          - n_filters: Number of filters to use
          - n_channels: Number of channels of the stimuli
          - device: Device to use. Defaults to 'cpu'
          - dtype: Data type to use. Defaults to torch.float32
        """
        super().__init__()
        self.n_dim = n_dim
        self.n_filters = n_filters
        self.register_buffer("priors", torch.as_tensor(priors))

        ### Make initial random filters
        filters = torch.randn(n_filters, n_channels, n_dim)
        # Model parameters
        self.filters = nn.Parameter(filters)
        register_parametrization(self, "filters", Sphere())


    #########################
    # PREPROCESSING
    #########################

    @abstractmethod
    def preprocess(self, stimuli):
        pass


    #########################
    # INFERENCE
    #########################

    @abstractmethod
    def responses(self, stimuli):
        """ Compute the response to each stimulus.

        -----------------
        Arguments:
        -----------------
            - stimuli: Stimulus tensor (n_stim x n_channels x n_dim)
        -----------------
        Output:
        -----------------
            - responses: Responses tensor (n_stim x n_filters)
        """
        pass


    def log_likelihoods(self, stimuli):
        """ Compute the log likelihood of each class for each stimulus.

        -----------------
        Arguments:
        -----------------
            - stimuli: Stimulus tensor (n_stim x n_channels x n_dim)
        -----------------
        Output:
        -----------------
            - log_likelihoods: Log-likelihoods tensor (n_stim x n_classes)
        """
        responses = self.responses(stimuli=stimuli)
        log_likelihoods = self.responses_2_log_likelihoods(responses)
        return log_likelihoods


    def posteriors(self, stimuli):
        """ Compute the posterior of each class for for each stimulus.

        -----------------
        Arguments:
        -----------------
            - stimuli: Stimulus tensor (n_stim x n_channels x n_dim)
        -----------------
        Output:
        -----------------
            - posteriors: Posteriors tensor (n_stim x n_classes)
        """
        log_likelihoods = self.log_likelihoods(stimuli=stimuli)
        posteriors = self.log_likelihoods_2_posteriors(log_likelihoods)
        return posteriors


    def estimates(self, stimuli):
        """ Compute latent variable estimates for each stimulus.

        -----------------
        Arguments:
        -----------------
            - stimuli: Stimulus tensor (n_stim x n_channels, n_dim)
        -----------------
        Output:
        -----------------
            - estimates: Estimates tensor (n_stim)
        """
        posteriors = self.posteriors(stimuli=stimuli)
        estimates = self.posteriors_2_estimates(posteriors=posteriors)
        return estimates


    @abstractmethod
    def responses_2_log_likelihoods(self, responses):
        """ Compute log-likelihood of each class given the filter responses.

        -----------------
        Arguments:
        -----------------
            - responses: Filter responses tensor (n_stim x n_filters)
        -----------------
        Output:
        -----------------
            - log_likelihoods: Log likelihoods tensor (n_stim x n_classes)
        """
        pass


    def log_likelihoods_2_posteriors(self, log_likelihoods):
        """ Compute the posterior of each class given the log likelihoods.

        -----------------
        Arguments:
        -----------------
            - log_likelihoods: Log likelihoods tensor (n_stim x n_classes)
        -----------------
        Output:
        -----------------
            - posteriors: Posteriors tensor (n_stim x n_classes)
        """
        posteriors = tfun.softmax(log_likelihoods + torch.log(self.priors), dim=-1)
        return posteriors


    def posteriors_2_estimates(self, posteriors):
        """ Convert posterior probabilities to estimates of the latent variable.

        -----------------
        Arguments:
        -----------------
            - posteriors: Posterior probabilities tensor (n_stim x n_classes)
        -----------------
        Output:
        -----------------
            - estimates: Vector with the estimated latent variable for each
                stimulus. (nStim)
        """
        # Get the index of the class with the highest posterior probability
        estimates = torch.argmax(posteriors, dim=-1)
        return estimates


    def forward(self, stimuli):
        """ Compute the class posteriors for the stimuli.
        -----------------
        Arguments:
        -----------------
            - stimuli: Stimulus tensor (n_stim x n_channels x n_dim)
        -----------------
        Output:
        -----------------
            - responses: Responses tensor (n_stim x n_filters)
        """
        posteriors = self.posteriors(stimuli)
        return posteriors


#####################
# AMA GAUSS
#####################

class AMAGauss(AMA):
    def __init__(self, stimuli, labels, n_filters=2, priors=None,
                 response_noise=0.0, c50=0.0, device='cpu', dtype=torch.float32):
        """
        -----------------
        AMA Gauss
        -----------------
        Assume that class-conditional responses are Gaussian distributed.

        -----------------
        Arguments:
        -----------------
            - stimuli: Stimulus tensor (n_stim x n_channels x n_dim)
            - labels: Label tensor (n_stim)
            - n_filters: Number of filters to use
            - priors: Prior probabilities of each class
        """
        # Initialize
        n_dim = stimuli.shape[-1]
        n_channels = stimuli.shape[-2]
        n_classes = torch.unique(labels).size()[0]

        if priors is None:
            priors = torch.ones(n_classes) / n_classes

        super().__init__(n_dim=stimuli.shape[-1], n_filters=n_filters, priors=priors,
                         n_channels=n_channels)
        self.register_buffer('c50', torch.as_tensor(c50))
        self.register_buffer('response_noise', torch.as_tensor(response_noise))

        ### Store stimuli statistics
        stimulus_statistics = inference.class_statistics(
          points=torch.flatten(self.preprocess(stimuli), -2, -1), # Collapse channels
          labels=labels
        )
        self.stimulus_statistics = BuffersDict(stimulus_statistics)


    def preprocess(self, stimuli):
        """ Divide each channel of each stimulus stimuli[i,c]
        by \sqrt{ ||stimuli[i,c]||^2 + c50} (square root of sum of squares + c50)
        -----------------
        Arguments:
        -----------------
            - stimuli: Stimulus tensor (n_stim x n_dim)
        -----------------
        Output:
        -----------------
            - stimuli_preprocessed: Processed stimuli tensor (nStim x n_dim)
        """
        return normalization.unit_norm_channels(stimuli, c50=self.c50)


    def responses(self, stimuli):
        """ Compute the responses of the filters to the stimuli
        (after pre-processing).
        -----------------
        Arguments:
        -----------------
            - stimuli: Stimulus tensor (n_stim x n_channels x n_dim)
        -----------------
        Output:
        -----------------
            - responses: Responses tensor (n_stim x n_filters)
        """
        stimuli_processed = self.preprocess(stimuli)
        responses = torch.einsum('kcd,ncd->nk', self.filters, stimuli_processed)
        return responses


    def responses_2_log_likelihoods(self, responses):
        """ Compute log-likelihood of each class given the filter responses.

        -----------------
        Arguments:
        -----------------
            - responses: Filter responses tensor (n_stim x n_filters)
        -----------------
        Output:
        -----------------
            - log_likelihoods: Log likelihoods tensor (n_stim x n_classes)
        """
        # Compute log likelihoods
        log_likelihoods = inference.gaussian_log_likelihoods(
          responses,
          self.response_statistics['means'],
          self.response_statistics['covariances']
        )
        return log_likelihoods


    @property
    def response_statistics(self):
        """Return the class-conditional response statistics.

        -----------------
        Output:
        -----------------
            - response_statistics: Dictionary with keys 'means' and 'covariances'
        """
        flat_filters = torch.flatten(self.filters, -2, -1)
        dtype = flat_filters.dtype
        device = flat_filters.device

        response_means = torch.einsum(
          'cd,kd->ck', self.stimulus_statistics['means'], flat_filters
        )

        noise_covariance = torch.eye(self.n_filters, dtype=dtype, device=device) * self.response_noise
        response_covariances = torch.einsum(
          'kd,cdb,mb->ckm', flat_filters, self.stimulus_statistics['covariances'], flat_filters
        )

        response_statistics = {
            'means': response_means,
            'covariances': response_covariances + noise_covariance
        }
        return response_statistics

    # Warn users that the response statistics can't be set
    @response_statistics.setter
    def response_statistics(self):
        raise AttributeError("The response statistics can't be set directly. "
                             "They are computed from the filters and the stimulus statistics.")


# Define the sphere constraint
class Sphere(nn.Module):
    def forward(self, X):
        """ Function to parametrize vectors on sphere.
        Each channel of X is first normalized to have unit norm,
        and then multiplied by the square root of the number of channels.
        -----------------
        Arguments:
        -----------------
            - X: Tensor in Euclidean space (n_filters x n_channels x n_dim)
        -----------------
        Output:
        -----------------
            - S: Tensor on sphere (n_filters x n_channels x n_dim)
        """
        return normalization.unit_norm(X)

    def right_inverse(self, S):
        """ Function to assign to parametrization"""
        return S

class BuffersDict(nn.Module):
    def __init__(self, stats_dict=None):
        super(BuffersDict, self).__init__()
        if stats_dict is not None:
            for name, tensor in stats_dict.items():
                self.register_buffer(name, tensor)

    def __getitem__(self, key):
        if key in self._buffers:
            return self._buffers[key]
        else:
            raise KeyError(f"'{key}' not found in BuffersDict")

    def __setitem__(self, key, value):
        self.register_buffer(key, value)

    def __delitem__(self, key):
        if key in self._buffers:
            del self._buffers[key]
        else:
            raise KeyError(f"'{key}' not found in BuffersDict")

    def __iter__(self):
        return iter(self._buffers)

    def __len__(self):
        return len(self._buffers)

    def keys(self):
        return self._buffers.keys()

    def items(self):
        return self._buffers.items()

    def values(self):
        return self._buffers.values()

    def __contains__(self, key):
        return key in self._buffers

    def __repr__(self):
        # Customize how tensors are represented
        def tensor_repr(tensor):
            if tensor.numel() > 10:
                # Show only the shape and dtype for large tensors
                return f"tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device})"
            else:
                # Use default tensor representation
                return repr(tensor)

        items_repr = ", ".join(
            f"'{key}': {tensor_repr(value)}"
            for key, value in self.items()
        )
        return f"BuffersDict({{{items_repr}}})"

    def __str__(self):
        return self.__repr__()
