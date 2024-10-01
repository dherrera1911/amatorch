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
    def __init__(self, n_dim, n_filters=2, n_channels=1, priors=1,
                 device='cpu', dtype=torch.float32):
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
        self.device = device
        self.dtype = dtype
        self.n_dim = n_dim
        self.n_filters = n_filters
        self.priors = priors

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
    def __init__(self, stimuli, labels, n_filters=2, values=None, c50=0.0, device='cpu',
                 dtype=torch.float32):
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
        super().__init__(n_dim=stimuli.shape[-1], n_filters=n_filters, n_channels=n_channels,
                         device=device, dtype=dtype)
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


#####################
#####################
# CHILD CLASS, EMPIRICAL
#####################
#####################


class AMA_emp(AMA):
    def __init__(self, sAll, ctgInd, n_filters=2, respNoiseVar=torch.tensor(0.02),
            pixelCov=torch.tensor(0), ctgVal=None, samplesPerStim=1, nChannels=1,
            device='cpu'):
        """
        -----------------
        Empirical AMA class
        -----------------
        This variant of AMA uses empirical estimates of the noisy normalized
        stimuli means and covariances.
        """
        ### Set basic attributes
        # Set device
        self.device = device
        # Set the number of channels, that are normalized separately
        self.nChannels = nChannels
        n_dim = sAll.shape[1]
        self.nClasses = torch.unique(ctgInd).size()[0]  # Number of classes
        # Initialize parent class
        super().__init__(n_dim=n_dim, n_filters=n_filters, device=device)

        # Make filter noise matrix
        self.respNoiseVar = torch.as_tensor(respNoiseVar)
        self.respNoiseVar.to(device)
        self.respNoiseCov = torch.eye(self.n_filtersAll, device=device) * self.respNoiseVar # DEVICE
        # Make random number generator for response noise
        self.respNoiseGen = MultivariateNormal(loc=torch.zeros(self.n_filtersAll, device=device),
                covariance_matrix=self.respNoiseCov)  # DEVICE

        ### Make noise generator
        # Turn noise parameters into tensors in device, and if needed convert
        # scalar into matrix
        pixelCov = torch.as_tensor(pixelCov)
        if pixelCov.dim()==0:
            self.pixelCov = torch.eye(sAll.shape[1], device=self.device) * pixelCov
        else:
            if pixelCov.shape[0] != sAll.shape[1]:
                raise ValueError('''Error: Stimulus noise covariance needs to
                        have the same dimensions as the stimuli''')
            self.pixelCov = pixelCov.to(self.device)
        # Make the noise generator for the stimuli
        self.stimNoiseGen = MultivariateNormal(loc=torch.zeros(sAll.shape[1],
                                                               device=self.device),
                                               covariance_matrix=self.pixelCov)
        ### Make noisy samples to compute statistics
        # Repeat sAll for samplesPerStim times along a new dimension
        sAllRep, ctgIndRep = au.repeat_stimuli(s=sAll, ctgInd=ctgInd, nReps=samplesPerStim)
        # Generate noise samples and add them to the repeated sAll tensor
        sProcessed = self.preprocess(s=sAllRep)
        ### Compute the conditional statistics of the stimuli
        self.stimMean = self.compute_norm_stim_mean(s=sProcessed, ctgInd=ctgIndRep)
        self.stimCov = self.compute_norm_stim_cov(s=sProcessed, ctgInd=ctgIndRep)
        ### Compute the conditional statistics of the responses
        self.respMean = self.compute_response_mean()
        self.respCovNoiseless = self.compute_response_cov()  # without filter noise
        self.respCov = self.respCovNoiseless + \
            self.respNoiseCov.repeat(self.nClasses, 1, 1)  # with filter noise


    def preprocess(self, s):
        """ Preprocess stimuli by adding noise and normalizing to
        unit norm.
        -----------------
        Arguments:
        -----------------
            - s: Stimulus matrix. (nStim x n_dim)
        -----------------
        Output:
        -----------------
            - sProcessed: Processed stimuli. (nStim x n_dim)
        """
        # Add noise to the stimuli
        noiseSamples = self.stimNoiseGen.sample([s.shape[0]])
        sNoisy = s + noiseSamples
        # Normalize the stimuli
        sProcessed = au.normalize_stimuli_channels(s=sNoisy, nChannels=self.nChannels)
        return sProcessed

      ### inference
    def get_responses(self, s, addRespNoise=True):
        """ Compute the responses of the filters to each stimulus in s. Note,
        stimuli are normalized to unit norm. If requested, noise is also
        added.
        -----------------
        Arguments:
        -----------------
            - s: stimulus matrix for which to compute posteriors. (nStim x n_dim)
            - addStimNoise: Logical that indicates whether to add noise to
                the input stimuli s. Added noise has the characteristics stored
                in the class.
        -----------------
        Output:
        -----------------
            - resp: Matrix with the filter responses for each stimulus.
            (nStim x n_filters)
        """
        # If given only vector as input, add singleton dimension
        if s.dim() == 1:
            s = s.unsqueeze(0)
        nStim = s.shape[0]
        # 1) Preprocess stimuli
        sProcessed = self.preprocess(s)
        # 2) Append fixed and trainable filters together
        fAll = self.all_filters()
        # 3) Apply filters to the stimuli
        resp = torch.einsum('fd,nd->nf', fAll, sProcessed)
        # 4) If requested, add response noise
        if addRespNoise:
            resp = resp + self.respNoiseGen.rsample([nStim])
        return resp



    ########################
    # STATISTICS COMPUTING
    ########################


    def compute_norm_stim_mean(self, s, ctgInd):
        """ Compute the mean of the noisy normalized stimuli across the
        dataset.
        -----------------
        Arguments:
        -----------------
            - s: Input PREPROCESED stimuli. (nStim x n_dim)
            - ctgInd: Category index of each stimulus. (nStim)
        -----------------
        Outputs:
        -----------------
            - stimMean: Mean of the noisy normalized stimuli for each category.
                (nClasses x n_dim)
        """
        # Compute the mean of noisy normalized stimuli for each category
        stimMean = au.category_means(s=s, ctgInd=ctgInd)
        return stimMean


    def compute_norm_stim_cov(self, s, ctgInd):
        """ Compute the covariance across the stimulus dataset for the noisy
        normalized stimuli. Uses some of the noise model properties
        stored as attributes of the class.
        -----------------
        Arguments:
        -----------------
            - s: Input PREPROCESSED stimuli. (nStim x n_dim)
            - ctgInd: Category index of each stimulus. (nStim)
        -----------------
        Outputs:
        -----------------
            - stimCov: Mean of the noisy normalized stimuli for each category.
                (nClasses x n_dim)
        """
        stimSecondM = au.category_secondM(s=s, ctgInd=ctgInd)
        stimCov = au.secondM_2_cov(secondM=stimSecondM, mean=self.stimMean)
        return stimCov


    def compute_response_mean(self):
        """ Compute the mean of the filter responses to the noisy stimuli
        for each class. Note that this are the means without added noise.
        -----------------
        Outputs:
        -----------------
            - respMean: Mean responses of each model filter to the noisy normalized
                stimuli of each class. (nClasses x n_filters)
        """
        fAll = self.all_filters()
        respMean = torch.einsum('cd,kd->ck', self.stimMean, fAll)
        return respMean


    def compute_response_cov(self):
        """ Compute the mean of the filter responses to the noisy stimuli for each class.
        Note that this are the noiseless filters.
        -----------------
        Outputs:
        -----------------
            - respCov: Covariance of filter responses to the noisy normalized
                stimuli of each class. (nClasses x n_filters x n_filters)
        """
        fAll = self.all_filters()
        ### Simplest method, only works for broadband normalization
        respCov = torch.einsum('kd,cdb,mb->ckm', fAll, self.stimCov, fAll)
        # Remove numerical errors that make asymmetric
        respCov = (respCov + respCov.transpose(1,2))/2
        return respCov


    def update_response_statistics(self):
        """ Update (in place) the conditional response means and covariances
        to match the current object filters
        """
        # Get all filters (fixed and trainable)
        fAll = self.all_filters()
        self.n_filtersTrain = self.filters.shape[0]
        self.n_filtersAll = fAll.shape[0]
        # If new filters were manually added, expand the response noise covariance
        self.respNoiseCov = torch.eye(self.n_filtersAll, device=self.device) * \
            self.respNoiseVar
        # Update covariances, size nClasses*n_filtersAll*n_filtersAll
        # Assign precomputed valeus, if same as initialization
        self.respMean = self.compute_response_mean()
        self.respCovNoiseless = self.compute_response_cov()
        # Add response noise to the stimulus-induced variability of responses
        self.respCov = self.respCovNoiseless + \
            self.respNoiseCov.repeat(self.nClasses, 1, 1)
        self.respNoiseGen = MultivariateNormal(
            loc=torch.zeros(self.n_filtersAll, device=self.device),
            covariance_matrix=self.respNoiseCov)


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
        nStim = resp.shape[0]
        # 1) Difference between responses and class means. (nStim x nClasses x n_filters)
        nMeans = self.respMean.shape[0]  # Don't use self.nClasses in case interpolated
        respDiff = resp.unsqueeze(1).repeat(1, nMeans, 1) - \
                self.respMean.unsqueeze(0).repeat(nStim, 1, 1)
        ## Get the log-likelihood of each class
        # 2) Quadratic component of log-likelihood (with negative sign)
        quadratics = -0.5 * torch.einsum('ncd,cdb,ncb->nc', respDiff,
                self.respCov.inverse(), respDiff)
        # 3) Constant term of log-likelihood
        llConst = -0.5 * self.n_filtersAll * torch.log(2*torch.tensor(torch.pi)) - \
            0.5 * torch.logdet(self.respCov)
        # 4) Add quadratics and constants to get log-likelihood
        ll = quadratics + llConst.repeat(nStim, 1)
        return ll


# Define the sphere constraint
class Sphere(nn.Module):
    def forward(self, X):
        """ Function to parametrize sphere vector S """
        # X is any vector
        S = X / torch.linalg.vector_norm(X, dim=1, keepdim=True) # Unit norm vector
        return S

    def right_inverse(self, S):
        """ Function to assign to parametrization""" 
        return S * S.shape[0]

class ClassStatistics(nn.Module):
    def __init__(self, stats_dict):
        super(ClassStatistics, self).__init__()
        for name, tensor in stats_dict.items():
            self.register_buffer(name, tensor)

