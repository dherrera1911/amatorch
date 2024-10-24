import torch

from amatorch import inference, normalization

from .ama_parent import AMAParent
from ._buffers_dict import BuffersDict


class AMAGauss(AMAParent):
    """
    AMAGauss model.

    This model assumes that class-conditional responses are Gaussian distributed.
    """

    def __init__(
        self,
        stimuli,
        labels,
        n_filters=None,
        filters=None,
        priors=None,
        response_noise=0.0,
        c50=0.0,
    ):
        """
        Initialize the AMAGauss model.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (n_stim, n_channels, n_dim).
        labels : torch.int64
            Label tensor of shape (n_stim).
        n_filters : int, optional
            Number of filters to use, by default 2.
        priors : torch.Tensor, optional
            Prior probabilities of each class, by default None.
        response_noise : float, optional
            Noise level in the responses, by default 0.0.
        c50 : float, optional
            Offset added to the denominator when normalizing stimuli,
            by default 0.0.
        """
        n_channels = stimuli.shape[-2]
        n_dim = stimuli.shape[-1]
        n_classes = torch.unique(labels).size()[0]

        if filters is not None:
            assert filters.shape[-2] == n_channels, "Channels of filters don't match stimuli."
            assert filters.shape[-1] == n_dim, "Dimensions of filters don't match stimuli."

        if priors is None:
            priors = torch.ones(n_classes) / n_classes

        super().__init__(
            priors=priors,
            n_dim=stimuli.shape[-1],
            n_filters=n_filters,
            n_channels=n_channels,
            filters=filters,
        )
        self.register_buffer("c50", torch.as_tensor(c50))
        self.register_buffer("response_noise", torch.as_tensor(response_noise))

        # Store stimuli statistics
        stimulus_statistics = inference.class_statistics(
            points=torch.flatten(self.preprocess(stimuli), -2, -1),  # Collapse channels
            labels=labels,
        )
        self.stimulus_statistics = BuffersDict(stimulus_statistics)

    def preprocess(self, stimuli):
        """
        Preprocess stimuli by normalizing each channel.

        Each channel of each stimulus is divided by the square root of the
        sum of squares plus `c50`.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (..., n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Processed stimuli tensor of shape (..., n_channels, n_dim).
        """
        return normalization.unit_norm_channels(stimuli, c50=self.c50)

    def get_responses(self, stimuli):
        """
        Compute the responses of the filters to the stimuli after
        pre-processing.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (..., n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Responses tensor of shape (..., n_filters).
        """
        stimuli_processed = self.preprocess(stimuli)
        responses = torch.einsum("kcd,...cd->...k", self.filters, stimuli_processed)
        return responses

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
        log_likelihoods = inference.gaussian_log_likelihoods(
            responses,
            self.response_statistics["means"],
            self.response_statistics["covariances"],
        )
        return log_likelihoods

    @property
    def response_statistics(self):
        """
        Return the class-conditional response statistics.

        Returns
        -------
        dict
            A dictionary containing:
            - 'means': torch.Tensor of shape (n_classes, n_filters).
            - 'covariances': torch.Tensor of shape (n_classes, n_filters, n_filters).
        """
        flat_filters = torch.flatten(self.filters, -2, -1)
        dtype = flat_filters.dtype
        device = flat_filters.device
        n_filters = flat_filters.shape[0]

        response_means = torch.einsum(
            "cd,kd->ck", self.stimulus_statistics["means"], flat_filters
        )

        noise_covariance = (
            torch.eye(n_filters, dtype=dtype, device=device) * self.response_noise
        )
        response_covariances = torch.einsum(
            "kd,cdb,mb->ckm",
            flat_filters,
            self.stimulus_statistics["covariances"],
            flat_filters,
        )

        response_statistics = {
            "means": response_means,
            "covariances": response_covariances + noise_covariance,
        }
        return response_statistics

    @response_statistics.setter
    def response_statistics(self):
        """
        Prevent direct setting of the response statistics.

        Raises
        ------
        AttributeError
            Raised if trying to set response statistics directly.
        """
        raise AttributeError(
            "Response statistics can't be set directly. "
            "They are computed from the filters and the "
            "stimulus statistics."
        )
