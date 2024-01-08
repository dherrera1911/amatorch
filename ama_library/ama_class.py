from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import einops as eo
from torch.distributions.multivariate_normal import MultivariateNormal
import geotorch
from ama_library import utilities as au
import qr_library as qr
import time

#####################
#####################
# PARENT AMA CLASS
#####################
#####################

class AMA(ABC, nn.Module):
    def __init__(self, nDim, nClasses, nFilt=2, respNoiseVar=torch.tensor(0.02),
            ctgVal=None, printWarnings=True, device='cpu'):
        """ AMA model object.
        -----------------
        Arguments:
        -----------------
          - sAll: Input stimuli. (nStim x nDim)
          - ctgInd: Category index of each stimulus. (nStim)
          - nFilt: Number of filters to train
          - respNoiseVar: Variance of filter response noise. Scalar
          - ctgVal: Value of the latent variable corresponding to each category.
              Defaults to equispaced points in [-1, 1].
              exact results. Defaults to 'post-filter'
          - printWarnings: Boolean indicating whether to print warnings.
          - device: Device to use. Defaults to 'cpu'
        -----------------
        Attributes:
        -----------------
          - f: Trainable filters. (nFiltTrain x nDim)
          - fFixed: Fixed filters. (nFiltFixed x nDim)
          - ctgVal: Value of the latent variable corresponding to each category
          - stimMean: Mean of the stimuli for each class
          - stimCov: Covariance of the stimuli for each class
          - respMean: Mean of the responses for each class
          - respCovNoiseless: Covariance of the responses for each class
          - respCov: Covariance of the responses for each class, including filter noise
          - respNoiseVar: Variance of filter response noise
          - respNoiseCov: Covariance of filter response noise
          - respNoiseGen: Response noise generator
          - nFiltTrain: Number of trainable filters
          - nFiltAll: Number of filters (trainable + fixed)
          - nDim: Number of dimensions of inputs
          - nClasses: Number of classes
          - printWarnings: Boolean indicating whether to print warnings
          - device: Device to use
        """
        super().__init__()
        self.printWarnings = printWarnings
        self.device = device
        ### Make initial random filters
        fInit = torch.randn(nFilt, nDim, device=device)  # DEVICE
        fInit = F.normalize(fInit, p=2, dim=1)
        # Model parameters
        self.f = nn.Parameter(fInit)
        geotorch.sphere(self, "f")
        # Attribute with fixed (non-trainable) filters
        self.fFixed = torch.tensor([], device=device)  # DEVICE
        # Assign handy variables
        self.nFiltTrain = self.f.shape[0]  # Number of trainable filters
        self.nFiltAll = self.nFiltTrain  # Number of filters including fixed filters
        self.nDim = self.f.shape[1]  # Number of dimensions
        self.nClasses = nClasses # Number of classes
        # If no category values given, assign equispaced values in [-1,1]
        if ctgVal is None:
            ctgVal = torch.linspace(start=-1, end=1, steps=self.nClasses)
        self.ctgVal = ctgVal.to(device)
        # Make filter noise matrix
        self.respNoiseVar = torch.as_tensor(respNoiseVar)
        self.respNoiseVar.to(device)
        self.respNoiseCov = torch.eye(self.nFiltAll, device=device) * self.respNoiseVar # DEVICE
        # Make random number generator for response noise
        self.respNoiseGen = MultivariateNormal(loc=torch.zeros(self.nFiltAll, device=device),
                covariance_matrix=self.respNoiseCov)  # DEVICE


    #########################
    ### BASIC UTILITY FUNCTIONS
    #########################


    def all_filters(self):
        """ Return a tensor with all the filters (fixed and trainable).
        Fixed filters are first.
        #
        -----------------
        Output:
        -----------------
            - fAll: Tensor with all filters. (nFiltAll x nDim)
        """
        return torch.cat((self.fFixed, self.f))


    def to(self, device):
        """ Move model tensors to the indicated  device. """
        super().to(device)
        self.device = device
        self.f = self.f.to(device)
        self.fFixed = self.fFixed.to(device)
        self.ctgVal = self.ctgVal.to(device)
        self.stimMean = self.stimMean.to(device)
        self.stimCov = self.stimCov.to(device)
        self.respMean = self.respMean.to(device)
        self.respCovNoiseless = self.respCovNoiseless.to(device)
        self.respCov = self.respCov.to(device)
        self.respNoiseCov = self.respNoiseCov.to(device)
        self.respNoiseGen = MultivariateNormal(loc=torch.zeros(self.nFiltAll, device=device),
                covariance_matrix=self.respNoiseCov)


    #########################
    ### FUNCTIONS FOR UPDATING MODEL STATISTICS
    #########################

    @abstractmethod
    def compute_norm_stim_mean(self, s, ctgInd):
        pass


    @abstractmethod
    def compute_norm_stim_cov(self, s, ctgInd):
        pass


    @abstractmethod
    def compute_response_mean(self):
        pass


    @abstractmethod
    def compute_response_cov(self):
        pass


    @abstractmethod
    def update_response_statistics(self):
        pass


    #########################
    ### FUNCTIONS FOR MODIFYING FILTERS
    #########################


    def assign_filter_values(self, fNew):
        """ Overwrite the values to the model filters.
        RESPONSE STATISTICS NEED TO BE UPDATED MANUALLY AFTER.
        -----------------
        Arguments:
        -----------------
            - fNew: Matrix with the new filters as rows. The new number of filters
                doesn't need to match the old number. (nFiltTrain x nDim)
        """
        # Remove parametrization so we can change the filters
        if parametrize.is_parametrized(self, "f"):
            parametrize.remove_parametrizations(self, "f", leave_parametrized=True)
        # Model parameters. Important to clone fNew, otherwise geotorch
        # modifies the original
        self.f = nn.Parameter(fNew.clone().to(self.device))
        geotorch.sphere(self, "f")
        self.f = fNew.to(self.device)
        # If number of trainable filters changed, update required params
        if self.nFiltTrain != fNew.shape[0]:
            self.nFiltTrain = fNew.shape[0]
            self.nFiltAll = self.nFiltTrain + self.fFixed.shape[0]
            self.respNoiseCov = torch.eye(self.nFiltAll, device=self.device) * \
                self.respNoiseVar
            self.respNoiseGen = MultivariateNormal(
                loc=torch.zeros(self.nFiltAll, device=self.device),
                covariance_matrix=self.respNoiseCov)


    def reinitialize_trainable(self):
        """ Re-initialize the trainable filters to random values.
        RESPONSE STATISTICS NEED TO BE UPDATED MANUALLY AFTER.
        """
        fRandom = torch.randn(self.nFiltTrain, self.nDim)
        fRandom = F.normalize(fRandom, p=2, dim=1)
        self.assign_filter_values(fNew=fRandom)


    def move_trainable_2_fixed(self):
        """ Set the trainable filters as fixed filters, and re-initialize
        the trainable filters to random values.
        RESPONSE STATISTICS NEED TO BE UPDATED MANUALLY AFTER.
        Input parameters are as in update_response_statistics()
        """
        newFix = self.all_filters().detach().clone()
        self.fFixed = newFix
        # reinitialize_trainable updates statistics
        self.reinitialize_trainable()


    def add_fixed_filters(self, fFixed):
        """ Add new filters to the model, that are not trainable parameters.
        -----------------
        Arguments:
        -----------------
            - fFixed: Te tensor with the new filters. (nFilt x nDim)
        """
        self.fFixed = fFixed.clone().to(self.device)


    def add_new_filters(self, nFiltNew):
        """ Add new, random filters to the filters already contained in
        the model.
        -----------------
        Arguments:
        -----------------
            - nFiltNew: number of new fiters to add
        """
        # Initialize new random filters and set length to 1 
        fNew = F.normalize(torch.randn(nFiltNew, self.nDim, device=self.device),
                           p=2, dim=1)
        fOld = self.f.detach().clone()
        f = torch.cat((fOld, fNew))  # Concatenate old and new filters
        self.assign_filter_values(fNew=f)


    #########################
    ### FUNCTIONS FOR GETTING MODEL OUTPUTS FOR INPUT STIMULI
    #########################


    @abstractmethod
    def preprocess(self, s):
        pass


    def get_responses(self, s, addRespNoise=True):
        """ Compute the responses of the filters to each stimulus in s. Note,
        stimuli are normalized to unit norm. If requested, noise is also
        added.
        -----------------
        Arguments:
        -----------------
            - s: stimulus matrix for which to compute posteriors. (nStim x nDim)
            - addStimNoise: Logical that indicates whether to add noise to
                the input stimuli s. Added noise has the characteristics stored
                in the class.
        -----------------
        Output:
        -----------------
            - resp: Matrix with the filter responses for each stimulus.
            (nStim x nFilt)
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


    def get_ll(self, s, addRespNoise=True):
        """ Compute the class posteriors for each stimulus in s. Noise
        can be added to the responses.
        -----------------
        Arguments:
        -----------------
            - s: stimulus matrix for which to compute posteriors. (nStim x nDim)
            - addRespNoise: Logical that indicates whether to add noise to the
                filter responses.
        -----------------
        Output:
        -----------------
            - ll: Matrix with the log-likelihood function across
            classes for each stimulus. (nStim x nClasses)
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        # 1) Get filter responses
        resp = self.get_responses(s=s, addRespNoise=addRespNoise)
        # 2) Get log-likelihood from the responses
        ll = self.resp_2_ll(resp)
        return ll


    def get_posteriors(self, s, addRespNoise=True):
        """ Compute the class posteriors for each stimulus in s.
        -----------------
        Arguments:
        -----------------
            - s: stimulus matrix for which to compute posteriors. (nStim x nDim)
            - addRespNoise: Logical that indicates whether to add noise to the
                filter responses.
        -----------------
        Output:
        -----------------
            - posteriors: Matrix with the posterior distribution across classes
            for each stimulus. (nStim x nClasses)
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        # 1) Get log-likelihoods
        ll = self.get_ll(s=s, addRespNoise=addRespNoise)
        # 2) Get posteriors from log-likelihoods
        posteriors = self.ll_2_posterior(ll)
        return posteriors


    def get_estimates(self, s, method4est='MAP', addRespNoise=True):
        """ Compute latent variable estimates for each stimulus in s.
        -----------------
        Arguments:
        -----------------
            - s: stimulus matrix for which to compute posteriors. (nStim x nDim)
            - method4est: Method to use for estimating the latent variable.
            - addRespNoise: Logical that indicates whether to add noise to the
                filter responses.
        -----------------
        Output:
        -----------------
            - estimates: Vector with the estimated latent variable for each
                stimulus. (nStim)
        """
        # 1) Compute posteriors from the stimuli
        posteriors = self.get_posteriors(s=s, addRespNoise=addRespNoise)
        # 2) Get estimates from the posteriors
        estimates = self.posterior_2_estimate(posteriors=posteriors,
                                              method4est=method4est)
        return estimates


    def resp_2_ll(self, resp):
        """ Compute log-likelihood of each class given the filter responses.
        -----------------
        Arguments:
        -----------------
            - resp: Matrix of filter responses. (nStim x nDim)
        -----------------
        Output:
        -----------------
            - ll: Matrix with the log-likelihood function across
            classes for each stimulus. (nStim x nClasses)
        """
        nStim = resp.shape[0]
        # 1) Difference between responses and class means. (nStim x nClasses x nFilt)
        nMeans = self.respMean.shape[0]  # Don't use self.nClasses in case interpolated
        respDiff = resp.unsqueeze(1).repeat(1, nMeans, 1) - \
                self.respMean.unsqueeze(0).repeat(nStim, 1, 1)
        ## Get the log-likelihood of each class
        # 2) Quadratic component of log-likelihood (with negative sign)
        quadratics = -0.5 * torch.einsum('ncd,cdb,ncb->nc', respDiff,
                self.respCov.inverse(), respDiff)
        # 3) Constant term of log-likelihood
        llConst = -0.5 * self.nFiltAll * torch.log(2*torch.tensor(torch.pi)) - \
            0.5 * torch.logdet(self.respCov)
        # 4) Add quadratics and constants to get log-likelihood
        ll = quadratics + llConst.repeat(nStim, 1)
        return ll


    def ll_2_posterior(self, ll):
        """ Convert log-likelihoods to posterior probabilities.
        -----------------
        Arguments:
        -----------------
            - ll: Matrix with the log-likelihood function across
            classes for each stimulus. (nStim x nClasses)
        -----------------
        Output:
        -----------------
            - posteriors: Matrix with the posterior distribution across classes
            for each stimulus. (nStim x nClasses)
        """
        posteriors = F.softmax(ll, dim=1)
        return posteriors


    def posterior_2_estimate(self, posteriors, method4est='MAP'):
        """ Convert posterior probabilities to estimates of the latent variable.
        -----------------
        Arguments:
        -----------------
            - posteriors: Matrix with the posterior distribution across classes
              for each stimulus. (nStim x nClasses)
            - method4est: Method to use for estimating the latent variable.
                Options are 'MAP' (maximum a posteriori) or 'MMSE' (minimum
                mean squared error).
        -----------------
        Output:
        -----------------
            - estimates: Vector with the estimated latent variable for each
                stimulus. (nStim)
        """
        if len(self.ctgVal) != posteriors.shape[1]:
            raise ValueError('''Error: ctgVal must have the same number of
                elements as the number of categories in the posteriors.''')
        if method4est=='MAP':
            # Get maximum posteriors indices of each stim, and its value
            (a, estimateInd) = torch.max(posteriors, dim=1)
            estimates = self.ctgVal[estimateInd]
        elif method4est=='MMSE':
            estimates = torch.einsum('nc,c->n', posteriors, self.ctgVal)
        return estimates


#####################
#####################
# CHILD CLASS, EMPIRICAL
#####################
#####################


class AMA_emp(AMA):
    def __init__(self, sAll, ctgInd, nFilt=2, respNoiseVar=torch.tensor(0.02),
            pixelCov=torch.tensor(0), ctgVal=None, samplesPerStim=1, nChannels=1,
            printWarnings=False, device='cpu'):
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
        nDim = sAll.shape[1]
        nClasses = torch.unique(ctgInd).size()[0]  # Number of classes
        # Initialize parent class
        super().__init__(nDim=nDim, nClasses=nClasses, nFilt=nFilt,
                respNoiseVar=respNoiseVar, ctgVal=ctgVal, printWarnings=printWarnings,
                device=device)
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
            - s: Stimulus matrix. (nStim x nDim)
        -----------------
        Output:
        -----------------
            - sProcessed: Processed stimuli. (nStim x nDim)
        """
        # Add noise to the stimuli
        noiseSamples = self.stimNoiseGen.sample([s.shape[0]])
        sNoisy = s + noiseSamples
        # Normalize the stimuli
        sProcessed = au.normalize_stimuli_channels(s=sNoisy, nChannels=self.nChannels)
        return sProcessed


    ########################
    # STATISTICS COMPUTING
    ########################


    def compute_norm_stim_mean(self, s, ctgInd):
        """ Compute the mean of the noisy normalized stimuli across the
        dataset.
        -----------------
        Arguments:
        -----------------
            - s: Input PREPROCESED stimuli. (nStim x nDim)
            - ctgInd: Category index of each stimulus. (nStim)
        -----------------
        Outputs:
        -----------------
            - stimMean: Mean of the noisy normalized stimuli for each category.
                (nClasses x nDim)
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
            - s: Input PREPROCESSED stimuli. (nStim x nDim)
            - ctgInd: Category index of each stimulus. (nStim)
        -----------------
        Outputs:
        -----------------
            - stimCov: Mean of the noisy normalized stimuli for each category.
                (nClasses x nDim)
        """
        stimSecondM = au.category_secondM(s=s, ctgInd=ctgInd)
        if self.printWarnings:
            print('''Warning: Response covariance updating is assuming
                    response means are already updated''')
        stimCov = au.secondM_2_cov(secondM=stimSecondM, mean=self.stimMean)
        return stimCov


    def compute_response_mean(self):
        """ Compute the mean of the filter responses to the noisy stimuli
        for each class. Note that this are the means without added noise.
        -----------------
        Outputs:
        -----------------
            - respMean: Mean responses of each model filter to the noisy normalized
                stimuli of each class. (nClasses x nFilt)
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
                stimuli of each class. (nClasses x nFilt x nFilt)
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
        self.nFiltTrain = self.f.shape[0]
        self.nFiltAll = fAll.shape[0]
        # If new filters were manually added, expand the response noise covariance
        self.respNoiseCov = torch.eye(self.nFiltAll, device=self.device) * \
            self.respNoiseVar
        # Update covariances, size nClasses*nFiltAll*nFiltAll
        # Assign precomputed valeus, if same as initialization
        self.respMean = self.compute_response_mean()
        self.respCovNoiseless = self.compute_response_cov()
        # Add response noise to the stimulus-induced variability of responses
        self.respCov = self.respCovNoiseless + \
            self.respNoiseCov.repeat(self.nClasses, 1, 1)
        self.respNoiseGen = MultivariateNormal(
            loc=torch.zeros(self.nFiltAll, device=self.device),
            covariance_matrix=self.respNoiseCov)


#####################
#####################
# CHILD CLASS, ISOTROPIC
#####################
#####################

class AMA_qmiso(AMA):
    def __init__(self, sAll, ctgInd, nFilt=2, respNoiseVar=torch.tensor(0.02),
            pixelVar=torch.tensor(0), ctgVal=None, printWarnings=False, device='cpu'):
        """
        -----------------
        Isotropic AMA class
        -----------------
        This variant of AMA adds isotropic noise to simuli and does normalization.
        It uses analytic formulas to estimate stimulus mean and covariance.
        """
        self.device = device
        # Set the number of channels, that are normalized separately
        nDim = sAll.shape[1]
        nClasses = torch.unique(ctgInd).size()[0]  # Number of classes
        # Initialize parent class
        super().__init__(nDim=nDim, nClasses=nClasses, nFilt=nFilt,
                respNoiseVar=respNoiseVar, ctgVal=ctgVal, printWarnings=printWarnings,
                device=device)
        # Make noise generator
        pixelVar = torch.as_tensor(pixelVar)
        self.pixelSigma = torch.sqrt(pixelVar).to(device)
        # Make the noise generator for the stimuli
        pixelCov = torch.eye(sAll.shape[1], device=self.device) * pixelVar
        self.stimNoiseGen = MultivariateNormal(loc=torch.zeros(sAll.shape[1],
                                                               device=self.device),
                                               covariance_matrix=pixelCov)
        ### Compute the conditional statistics of the stimuli
        # CHECK THAT THESE FUNCTIONS WORK
        self.stimMean = self.compute_norm_stim_mean(s=sAll, ctgInd=ctgInd)
        self.stimCov = self.compute_norm_stim_cov(s=sAll, ctgInd=ctgInd)
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
            - s: Stimulus matrix. (nStim x nDim)
        -----------------
        Output:
        -----------------
            - sProcessed: Processed stimuli. (nStim x nDim)
        """
        # Add noise to the stimuli
        noiseSamples = self.stimNoiseGen.sample([s.shape[0]])
        sNoisy = s + noiseSamples
        # Normalize the stimuli
        sProcessed = au.normalize_stimuli_channels(s=sNoisy)
        return sProcessed


    ########################
    # STATISTICS COMPUTING
    ########################


    def compute_norm_stim_mean(self, s, ctgInd):
        """ Compute the mean of the noisy normalized stimuli across the
        dataset.
        -----------------
        Arguments:
        -----------------
            - s: Input stimuli. (nStim x nDim)
            - nc: Non-centrality parameter (nStim)
            - ctgInd: Category index of each stimulus. (nStim)
        -----------------
        Outputs:
        -----------------
            - stimMean: Mean of the noisy normalized stimuli for each category.
                (nClasses x nDim)
        """
        nClasses = torch.unique(ctgInd).size()[0]
        sCtgMean = torch.zeros(nClasses, self.nDim, device=s.device)
        for cl in range(nClasses):
            mask = (ctgInd == cl)
            sLevel = s[mask,:]  # Stimuli of the same category
            nStim = sLevel.size()[0]
            # CHECK THIS IS RUNNING CORRECTLY 28/12/2023
            sMean = qr.projected_normal_mean_iso(mu=s, sigma=self.pixelSigma)
            sCtgMean[cl,:] = eo.reduce(sMean, 'n b -> b', 'mean')
        return sCtgMean


    def compute_norm_stim_cov(self, s, ctgInd, sameAsInit=True):
        """ Compute the covariance across the stimulus dataset for the noisy
        normalized stimuli. Uses some of the noise model properties
        stored as attributes of the class.
        -----------------
        Arguments:
        -----------------
            - s: Input stimuli. (nStim x nDim)
            - ctgInd: Category index of each stimulus. (nStim)
        -----------------
        Outputs:
        -----------------
            - stimCov: Covariance matrices of the noisy normalized stimuli
                for each category. (nClasses x nDim x nDim)
        """
        nClasses = torch.unique(ctgInd).size()[0]
        sCtgCov = torch.zeros(nClasses, self.nDim, self.nDim, device=s.device)
        for cl in range(nClasses):
            mask = (ctgInd == cl)
            sLevel = s[mask,:]  # Stimuli of the same category
            # CHECK THIS IS RUNNING CORRECTLY 28/12/2023
            sm = qr.projected_normal_sm_iso_batch(mu=sLevel, sigma=self.pixelSigma)
            sCtgCov[cl,:,:] = qr.secondM_2_cov(secondM=sm,
                                              mean=self.stimMean[cl,:])
        return sCtgCov


    def compute_response_mean(self):
        """ Compute the mean of the filter responses to the noisy stimuli
        for each class. Note that this are the means without added noise.
        -----------------
        Outputs:
        -----------------
            - respMean: Mean responses of each model filter to the noisy normalized
                stimuli of each class. (nClasses x nFilt)
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
                stimuli of each class. (nClasses x nFilt x nFilt)
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
        self.nFiltTrain = self.f.shape[0]
        self.nFiltAll = fAll.shape[0]
        # If new filters were manually added, expand the response noise covariance
        self.respNoiseCov = torch.eye(self.nFiltAll, device=self.device) * \
            self.respNoiseVar
        # Update covariances, size nClasses*nFiltAll*nFiltAll
        # Assign precomputed valeus, if same as initialization
        self.respMean = self.compute_response_mean()
        self.respCovNoiseless = self.compute_response_cov()
        # Add response noise to the stimulus-induced variability of responses
        self.respCov = self.respCovNoiseless + \
            self.respNoiseCov.repeat(self.nClasses, 1, 1)
        self.respNoiseGen = MultivariateNormal(
            loc=torch.zeros(self.nFiltAll, device=self.device),
            covariance_matrix=self.respNoiseCov)

