from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as tfun
from torch.nn.utils.parametrize import register_parametrization

from amatorch import constraints


class AMAParent(ABC, nn.Module):
    """
    Abstract AMA parent class.
    """

    def __init__(self, n_dim, n_filters, priors, n_channels=1):
        """
        Initialize the AMA model.

        Parameters
        ----------
        n_dim : int
            Number of dimensions of inputs.
        n_filters : int
            Number of filters to use.
        n_channels : int, optional
            Number of channels of the stimuli, by default 1.
        priors : torch.Tensor
            Prior probabilities for each class.
        """
        super().__init__()
        self.n_dim = n_dim
        self.register_buffer("priors", torch.as_tensor(priors))

        # Make initial random filters
        filters = torch.randn(n_filters, n_channels, n_dim)
        # Model parameters
        self.filters = nn.Parameter(filters)
        register_parametrization(self, "filters", constraints.Sphere())

    #########################
    # PREPROCESSING
    #########################

    @abstractmethod
    def preprocess(self, stimuli):
        """
        Preprocess the stimuli before computing the responses.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (n_stim, n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Preprocessed stimuli of shape (n_stim, n_channels, n_dim).
        """
        pass

    #########################
    # INFERENCE
    #########################

    @abstractmethod
    def responses(self, stimuli):
        """
        Compute the response to each stimulus.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (n_stim, n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Responses tensor of shape (n_stim, n_filters).
        """
        pass

    def log_likelihoods(self, stimuli):
        """
        Compute the log-likelihood of each class for each stimulus.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (n_stim, n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Log-likelihoods tensor of shape (n_stim, n_classes).
        """
        responses = self.responses(stimuli=stimuli)
        log_likelihoods = self.responses_2_log_likelihoods(responses)
        return log_likelihoods

    def posteriors(self, stimuli):
        """
        Compute the posterior of each class for each stimulus.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (n_stim, n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Posteriors tensor of shape (n_stim, n_classes).
        """
        log_likelihoods = self.log_likelihoods(stimuli=stimuli)
        posteriors = self.log_likelihoods_2_posteriors(log_likelihoods)
        return posteriors

    def estimates(self, stimuli):
        """
        Compute latent variable estimates for each stimulus.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (n_stim, n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Estimates tensor of shape (n_stim).
        """
        posteriors = self.posteriors(stimuli=stimuli)
        estimates = self.posteriors_2_estimates(posteriors=posteriors)
        return estimates

    @abstractmethod
    def responses_2_log_likelihoods(self, responses):
        """
        Compute log-likelihood of each class given the filter responses.

        Parameters
        ----------
        responses : torch.Tensor
            Filter responses tensor of shape (n_stim, n_filters).

        Returns
        -------
        torch.Tensor
            Log-likelihoods tensor of shape (n_stim, n_classes).
        """
        pass

    def log_likelihoods_2_posteriors(self, log_likelihoods):
        """
        Compute the posterior of each class given the log-likelihoods.

        Parameters
        ----------
        log_likelihoods : torch.Tensor
            Log-likelihoods tensor of shape (n_stim, n_classes).

        Returns
        -------
        torch.Tensor
            Posteriors tensor of shape (n_stim, n_classes).
        """
        posteriors = tfun.softmax(log_likelihoods + torch.log(self.priors), dim=-1)
        return posteriors

    def posteriors_2_estimates(self, posteriors):
        """
        Convert posterior probabilities to estimates of the latent variable.

        Parameters
        ----------
        posteriors : torch.Tensor
            Posterior probabilities tensor of shape (n_stim, n_classes).

        Returns
        -------
        torch.Tensor
            Estimates tensor of shape (n_stim), containing the estimated latent
            variable for each stimulus.
        """
        # Get the index of the class with the highest posterior probability
        estimates = torch.argmax(posteriors, dim=-1)
        return estimates

    def forward(self, stimuli):
        """
        Compute the class posteriors for the stimuli.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (n_stim, n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Posteriors tensor of shape (n_stim, n_classes).
        """
        posteriors = self.posteriors(stimuli)
        return posteriors
