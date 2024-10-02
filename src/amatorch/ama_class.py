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
        return self.responses_2_log_likelihoods(responses)


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
        return self.log_likelihoods_2_posteriors(log_likelihoods)


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
        return self.posteriors_2_estimates(posteriors=posteriors)


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
        return tfun.softmax(log_likelihoods + torch.log(self.posteriors), dim=-1)


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
        # Get maximum posteriors indices of each stim, and its value
        (a, labels) = torch.max(posteriors, dim=-1)
        return labels


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
        return self.posteriors(stimuli)


    #########################
    # RESPONSE DISTRIBUTIONS
    #########################

    @abstractmethod
    def update_response_distributions(self):
        pass



#####################
# AMA GAUSS
#####################

class AMAGauss(AMA):
    def __init__(self, stimuli, labels, n_filters=2, priors=None,
                 c50=0.0, device='cpu', dtype=torch.float32):
        """
        -----------------
        AMA Gauss
        -----------------
        Assume that class-conditional responses are Gaussian distributed.
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

        ### Compute stimuli statistics
        stimulus_statistics = inference.class_statistics(
          points=torch.flatten(self.preprocess(stimuli), -2, -1), # Collapse channels
          labels=labels
        )
        self.stimulus_statistics = ClassStatistics(stimulus_statistics)

        ### Compute response statistics
        response_statistics = {
          'means': torch.zeros(n_classes, n_filters),
          'covariances': torch.eye(n_filters).repeat(n_classes, 1, 1)
        }
        self.response_statistics = ClassStatistics(response_statistics)
        self.update_response_distributions()


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
        return normalization.unit_norm(stimuli, c50=self.c50)


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
        return inference.gaussian_log_likelihoods(
          responses,
          self.response_statistics.means,
          self.response_statistics.covariances
        )


    def update_response_distributions(self):
        """
        Update the class-conditional response means and covariances
        """
        # Update means
        flat_filters = torch.flatten(self.filters, -2, -1)
        response_means = torch.einsum(
          'cd,kd->ck', self.stimulus_statistics.means, flat_filters
        )
        self.response_statistics.means.resize_(response_means.shape)
        self.response_statistics.means.copy_(response_means)

        # Update covariances
        response_covariances = torch.einsum(
          'kd,cdb,mb->ckm', flat_filters, self.stimulus_statistics.covariances,
          flat_filters
        )
        self.response_statistics.covariances.resize_(response_covariances.shape)
        self.response_statistics.covariances.copy_(response_covariances)


# Define the sphere constraint
class Sphere(nn.Module):
    def forward(self, X):
        """ Function to parametrize sphere vector S """
        # X is any vector
        S = X / torch.linalg.vector_norm(X, dim=-1, keepdim=True) # Unit norm vector
        return S

    def right_inverse(self, S):
        """ Function to assign to parametrization""" 
        return S * S.shape[0]

class ClassStatistics(nn.Module):
    def __init__(self, stats_dict):
        super(ClassStatistics, self).__init__()
        for name, tensor in stats_dict.items():
            self.register_buffer(name, tensor)

