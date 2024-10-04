from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as tfun
from torch.nn.utils.parametrize import register_parametrization
from amatorch import constraints


class AMAParent(ABC, nn.Module):
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
        register_parametrization(self, "filters", constraints.Sphere())


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
