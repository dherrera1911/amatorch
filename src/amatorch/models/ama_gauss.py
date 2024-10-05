import torch

from amatorch import inference, normalization

from .ama_parent import AMAParent
from .buffers_dict import BuffersDict


class AMAGauss(AMAParent):
    def __init__(
        self,
        stimuli,
        labels,
        n_filters=2,
        priors=None,
        response_noise=0.0,
        c50=0.0,
        device="cpu",
        dtype=torch.float32,
    ):
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
        n_channels = stimuli.shape[-2]
        n_classes = torch.unique(labels).size()[0]

        if priors is None:
            priors = torch.ones(n_classes) / n_classes

        super().__init__(
            n_dim=stimuli.shape[-1],
            n_filters=n_filters,
            priors=priors,
            n_channels=n_channels,
        )
        self.register_buffer("c50", torch.as_tensor(c50))
        self.register_buffer("response_noise", torch.as_tensor(response_noise))

        ### Store stimuli statistics
        stimulus_statistics = inference.class_statistics(
            points=torch.flatten(self.preprocess(stimuli), -2, -1),  # Collapse channels
            labels=labels,
        )
        self.stimulus_statistics = BuffersDict(stimulus_statistics)

    def preprocess(self, stimuli):
        """Divide each channel of each stimulus stimuli[i,c]
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
        """Compute the responses of the filters to the stimuli
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
        responses = torch.einsum("kcd,ncd->nk", self.filters, stimuli_processed)
        return responses

    def responses_2_log_likelihoods(self, responses):
        """Compute log-likelihood of each class given the filter responses.

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
            self.response_statistics["means"],
            self.response_statistics["covariances"],
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
            "cd,kd->ck", self.stimulus_statistics["means"], flat_filters
        )

        noise_covariance = (
            torch.eye(self.n_filters, dtype=dtype, device=device) * self.response_noise
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

    # Warn users that the response statistics can't be set
    @response_statistics.setter
    def response_statistics(self):
        raise AttributeError(
            "Response statistics can't be set directly. "
            "They are computed from the filters and the "
            "stimulus statistics."
        )
