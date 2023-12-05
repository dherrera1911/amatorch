from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import geotorch
import torch.nn.utils.parametrize as parametrize
from torch.distributions.multivariate_normal import MultivariateNormal
from ama_library import utilities as au
from ama_library import quadratic_moments as qm
from ama_library import geometry as ag
import time

#####################
#####################
# PARENT AMA CLASS
#####################
#####################

class AMA(ABC, nn.Module):
    def __init__(self, sAll, ctgInd, nFilt=2, respNoiseVar=torch.tensor(0.02),
            ctgVal=None, filtNorm='broadband', respCovPooling='pre-filter',
            printWarnings=True, device='cpu'):
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
          - filtNorm: String indicating method use to normalize filter responses.
              Can be 'broadband' (default) or 'narrowband'
          - respCovPooling: String indicating how to compute the filter response
              covariances. 'pre-filter' can be less accurate depending on the
              problem, but is faster. 'post-filter' is slower but gives
              exact results. Defaults to 'post-filter'
          - printWarnings: Boolean indicating whether to print warnings.
          - device: Device to use. Defaults to 'cpu'
        -----------------
        Attributes:
        -----------------
          - f: Trainable filters. (nFilt x nDim)
          - fFixed: Fixed filters. (nFiltFixed x nDim)
          - ctgVal: Value of the latent variable corresponding to each category
          - stimMean: Mean of the stimuli for each class
          - stimCov: Covariance of the stimuli for each class
          - respMean: Mean of the responses for each class
          - respCovNoiseless: Covariance of the responses for each class
          - respCov: Covariance of the responses for each class, including filter noise
          - respNoiseVar: Variance of filter response noise
          - respNoiseCov: Covariance of filter response noise
          - stimNoiseGen: Pixel noise generator
          - respNoiseGen: Response noise generator
          - nFilt: Number of trainable filters
          - nFiltAll: Number of filters (trainable + fixed)
          - nDim: Number of dimensions of inputs
          - nClasses: Number of classes
          - filtNorm: Method to normalize filter responses
          - respCovPooling: Method to compute filter response covariances
          - printWarnings: Boolean indicating whether to print warnings
          - device: Device to use
          - pixelCov: Covariance of the pixel noise
        """
        super().__init__()
        print('Initializing AMA')
        self.printWarnings = printWarnings
        self.device = device
        ### Make initial random filters
        fInit = torch.randn(nFilt, sAll.shape[1], device=device)  # DEVICE
        fInit = F.normalize(fInit, p=2, dim=1)
        # Model parameters
        self.f = nn.Parameter(fInit)
        geotorch.sphere(self, "f")
        # Attribute with fixed (non-trainable) filters
        self.fFixed = torch.tensor([], device=device)  # DEVICE
        # Assign handy variables
        self.nFilt = self.f.shape[0]  # Number of trainable filters
        self.nFiltAll = self.nFilt  # Number of filters including fixed filters
        self.nDim = self.f.shape[1]  # Number of dimensions
        self.nClasses = torch.unique(ctgInd).size()[0]  # Number of classes
        self.filtNorm = filtNorm  # Method to normalize the filters
        self.respCovPooling = respCovPooling  # Method to generate response covariance
        # If no category values given, assign equispaced values in [-1,1]
        if ctgVal is None:
            ctgVal = torch.linspace(start=-1, end=1, steps=self.nClasses)
        self.ctgVal = ctgVal.to(device)
        # Make filter noise matrix (##IMPLEMENT GENERAL NOISE COVARIANCE LATER##)
        self.respNoiseVar = torch.tensor(respNoiseVar, device=device)
        self.respNoiseCov = torch.eye(self.nFiltAll, device=device) * self.respNoiseVar # DEVICE
        # Make random number generators for pixel and response noise
        self.stimNoiseGen = MultivariateNormal(loc=torch.zeros(self.nDim, device=device),
                covariance_matrix=self.pixelCov)  # DEVICE
        self.respNoiseGen = MultivariateNormal(loc=torch.zeros(self.nFiltAll, device=device),
                covariance_matrix=self.respNoiseCov)  # DEVICE
        ### Compute the conditional statistics of the stimuli
        self.stimMean = self.compute_norm_stim_mean(s=sAll, ctgInd=ctgInd)
        self.stimCov = self.compute_norm_stim_cov(s=sAll, ctgInd=ctgInd)
        ### Compute the conditional statistics of the responses
        self.respMean = self.compute_response_mean(s=sAll, ctgInd=ctgInd)
        self.respCovNoiseless = self.compute_response_cov(s=sAll, ctgInd=ctgInd)  # without filter noise
        self.respCov = self.respCovNoiseless + \
            self.respNoiseCov.repeat(self.nClasses, 1, 1)  # with filter noise


    #########################
    ### BASIC UTILITY FUNCTIONS
    #########################

    def fixed_and_trainable_filters(self):
        """ Return a tensor with all the filters (fixed and trainable).
        Fixed filters are first.
        #
        -----------------
        Output:
        -----------------
            - fAll: Tensor with all filters. (nFilt x nDim)
        """
        return torch.cat((self.fFixed, self.f))


    def make_noisy_normalized_stimuli(self, s, ctgInd=None, samplesPerStim=1):
        """ Generate noisy stimuli samples and normalize them.
        # 
        -----------------
        Arguments:
        -----------------
          - s: Input noiseless stimuli. (nStim x nDim)
          - ctgInd: Category index of each stimulus. (nStim)
          - samplesPerStim: Number of noisy samples to generate per stimulus.
        -----------------
        Output:
        -----------------
          - sAllNoisy: Noisy stimuli. (nStim x nDim)
          - ctgIndNoisy: Category index of each noisy stimulus. (nStim)
        """
        # Generate noisy stimuli samples
        n, d = s.shape
        # Repeat stimuli for samplesPerStim times along a new dimension
        sRepeated = s.repeat(samplesPerStim, 1, 1)
        # Generate noise samples and add them to the repeated s tensor
        noiseSamples = self.stimNoiseGen.sample((samplesPerStim, n))
        sAllNoisy = sRepeated + noiseSamples
        sAllNoisy = sAllNoisy.transpose(0, 1).reshape(-1, d)
        if not ctgInd is None:
            ctgIndNoisy = ctgInd.repeat_interleave(samplesPerStim)
        else:
            ctgIndNoisy = None
        # Normalize the stimuli
        sAllNoisy = au.normalize_stimuli_channels(s=sAllNoisy,
                nChannels=self.nChannels)
        return sAllNoisy, ctgIndNoisy


    def to(self, device):
        """ Move model tensors to the indicated  device. """
        super().to(device)
        self.device = device
        self.fFixed = self.fFixed.to(device)
        self.ctgVal = self.ctgVal.to(device)
        self.stimMean = self.stimMean.to(device)
        self.stimCov = self.stimCov.to(device)
        self.respMean = self.respMean.to(device)
        self.respCovNoiseless = self.respCovNoiseless.to(device)
        self.respCov = self.respCov.to(device)
        self.respNoiseCov = self.respNoiseCov.to(device)
        self.pixelCov = self.pixelCov.to(device)
        self.respNoiseGen = MultivariateNormal(loc=torch.zeros(self.nFiltAll, device=device),
                covariance_matrix=self.respNoiseCov)
        self.stimNoiseGen = MultivariateNormal(loc=torch.zeros(self.nDim, device=device),
                covariance_matrix=self.pixelCov)
        if self.amaType == 'Isotropic':
            self.smNoiseW = self.smNoiseW.to(device)
            self.smMeanW = self.smMeanW.to(device)
            self.nonCentrality = self.nonCentrality.to(device)
            self.invNormMean = self.invNormMean.to(device)
            self.pixelSigma = self.pixelSigma.to(device)


    #########################
    ### FUNCTIONS FOR UPDATING MODEL STATISTICS AND FILTERS
    #########################

    # Methods for updating model statistics depend on the procedure to
    # be used, and on the model's architecture. Define abstract methods
    # that are implemented in the child classes.

    @abstractmethod
    def compute_norm_stim_mean(self, s, ctgInd):
        pass


    @abstractmethod
    def compute_norm_stim_cov(self, s, ctgInd):
        pass


    @abstractmethod
    def compute_response_mean(self, s, ctgInd):
        pass


    @abstractmethod
    def compute_response_cov(self, s, ctgInd):
        pass


#    @abstractmethod
#    def compute_normalized_stimulus_amplitude_spec(self, s, **kwargs):
#        pass


    def update_response_statistics(self, sAll, ctgInd, sAmp=None, sameAsInit=True):
        """ Update (in place) the conditional response means and covariances
        to match the current object filters
        -----------------
        Arguments:
        -----------------
            - sAll: Stimulus dataset to use for response updating. Ideally, it is
             the same dataset used to initialize the model, for which there are
             precomputed quantities. (nStim x nDim)
            - ctgInd: Vector with category indices of the stimuli
            - sAmp: Optional to save compute. Pre-computed amplitude spectrum
            of the stimulus dataset. Should be computed
            with 'au.compute_amplitude_spectrum'. (nStim x nDim)
            - sameAsInit: Logical indicating whether sAll is the same dataset
            used for initialization. In that case, precomputed values are used
            to get the statistics
        """
        # Get all filters (fixed and trainable)
        fAll = self.fixed_and_trainable_filters()
        self.nFilt = self.f.shape[0]
        self.nFiltAll = fAll.shape[0]
        # If new filters were manually added, expand the response noise covariance
        ### NEED TO MODIFY FOR NON ISOTROPIC NOISE
        self.respNoiseCov = torch.eye(self.nFiltAll, device=self.device) * \
            self.respNoiseVar
        # Update covariances, size nClasses*nFilt*nFilt
        # Assign precomputed valeus, if same as initialization
        self.respMean = self.compute_response_mean(s=sAll, ctgInd=ctgInd, sAmp=sAmp)
        self.respCovNoiseless = self.compute_response_cov(s=sAll, ctgInd=ctgInd,
              sAmp=sAmp)
        if self.amaType == 'Isotropic' and self.printWarnings:
            print('''Warning: Response statistics update assuming that stimuli
                    for updating and for initialization are the same''')
        # Add response noise to the stimulus-induced variability of responses
        self.respCov = self.respCovNoiseless + \
            self.respNoiseCov.repeat(self.nClasses, 1, 1)
        self.respNoiseGen = MultivariateNormal(
            loc=torch.zeros(self.nFiltAll, device=self.device),
            covariance_matrix=self.respNoiseCov)
        # If statistics were interpolated, return ctgVal to original length
        if not self.ctgVal.shape[0] == self.nClasses:
            subsampleFactor = int(len(self.ctgVal) / self.nClasses + 1)
            if len(self.ctgVal) % 2 == 1:
                subsampleInds = au.subsample_categories_centered(nCtg=len(self.ctgVal),
                                                      subsampleFactor=subsampleFactor)
            else:
                # CHECK THAT THIS WORKS
                subsampleInds = np.arange(0, len(self.ctgVal), subsampleFactor)
            self.ctgVal = self.ctgVal[subsampleInds]
            self.nClasses = len(self.respCov)


    #### Consider making this child-specific
    def assign_filter_values(self, fNew, sAll, ctgInd, sAmp=None,
            sameAsInit=True):
        """ Overwrite the values to the model filters. Updates model
        parameters and statistics accordingly.
        -----------------
        Arguments:
        -----------------
            - fNew: Matrix with the new filters as rows. The new number of filters
                doesn't need to match the old number. (nFilt x nDim)
            - Rest of inputs, sAll, ctgInd, sAmp, sameAsInit, are as explained
                in update_response_statistics()
        """
        # Remove parametrization so we can change the filters
        if parametrize.is_parametrized(self, "f"):
            parametrize.remove_parametrizations(self, "f", leave_parametrized=True)
        # Model parameters. Important to clone fNew, otherwise geotorch
        # modifies the original
        self.f = nn.Parameter(fNew.clone().to(self.device))
        geotorch.sphere(self, "f")
        self.f = fNew.to(self.device)
        # Update model values
        self.update_response_statistics(sAll=sAll, ctgInd=ctgInd, sAmp=sAmp,
                sameAsInit=sameAsInit)
        # In case new filters were added, adapt the noise generator
        self.respNoiseGen = MultivariateNormal(
            loc=torch.zeros(self.nFiltAll, device=self.device),
            covariance_matrix=self.respNoiseCov)


    def reinitialize_trainable(self, sAll, ctgInd, sAmp=None, sameAsInit=True):
        """ Re-initialize the trainable filters to random values.
        Input parameters are as in update_response_statistics()
        """
        fRandom = torch.randn(self.nFilt, self.nDim)
        fRandom = F.normalize(fRandom, p=2, dim=1)
        # Statistics are updated in assign_filter_values
        self.assign_filter_values(fNew=fRandom, sAll=sAll, ctgInd=ctgInd,
                sAmp=sAmp, sameAsInit=sameAsInit)


    def move_trainable_2_fixed(self, sAll, ctgInd, sAmp=None, sameAsInit=True):
        """ Set the trainable filters as fixed filters, and re-initialize
        the trainable filters to random values.
        Input parameters are as in update_response_statistics()
        """
        newFix = self.fixed_and_trainable_filters().detach().clone()
        self.fFixed = newFix
        # reinitialize_trainable updates statistics
        self.reinitialize_trainable(sAll=sAll, ctgInd=ctgInd, sAmp=sAmp,
                sameAsInit=sameAsInit)


    def add_fixed_filters(self, fFixed, sAll, ctgInd, sAmp=None, sameAsInit=True):
        """ Add new filters to the model, that are not trainable parameters.
        -----------------
        Arguments:
        -----------------
            - fFixed: Te tensor with the new filters. (nFilt x nDim)
            - Rest of input parameters are as in update_response_statistics()
        """
        self.fFixed = fFixed.clone().to(self.device)
        self.update_response_statistics(sAll=sAll, ctgInd=ctgInd, sAmp=None,
                sameAsInit=True)


    def add_new_filters(self, nFiltNew, sAll, ctgInd, sAmp=None,
            sameAsInit=True):
        """ Add new, random filters to the filters already contained in
        the model. Adapt model statistics and parameters accordingly.
        -----------------
        Arguments:
        -----------------
            - nFiltNew: number of new fiters to add
            - Rest of inputs, sAll, ctgInd, sAmp, sameAsInit, are as explained
               in update_response_statistics()
        """
        # Initialize new random filters and set length to 1 
        fNew = F.normalize(torch.randn(nFiltNew, self.nDim, device=self.device),
                           p=2, dim=1)
        fOld = self.f.detach().clone()
        f = torch.cat((fOld, fNew))  # Concatenate old and new filters
        # assign_filter_values updates statistics
        self.assign_filter_values(fNew=f, sAll=sAll, ctgInd=ctgInd, sAmp=sAmp,
                sameAsInit=sameAsInit)


    def interpolate_class_statistics(self, nPoints=11, method='geodesic',
                                     metric='BuressWasserstein',
                                     variableType='linear',
                                     circularUnits='deg'):
        """
        Add new classes to the model, by interpolating between the existing classes.
        The size of self.respCov, self.respMean and self.ctgVal are changed to
        incorporate more (interpolated) classes. Doesn't update attribute
        nClasses (which stays indicating the number of original classes).
        -----------------
        Arguments:
        -----------------
            - nPoints: number of points to interpolate between available points
            - method: method used for interpolation ('geodesic' or 'spline')
            - metric: metric used for interpolation (e.g., 'BuressWasserstein')
            - variableType: type of variable, can be 'linear' or 'circular'
            - circularUnits: units of circular variable, can be 'deg' or 'rad'
        """
        # Handle circular variables
        if variableType == 'circular':
            # Triplicate the means and covariances for 3 full turns around the circle
            respMean = torch.cat([self.respMean.detach()] * 3, dim=0)
            respCov = torch.cat([self.respCov.detach()] * 3, dim=0)
            ctgVal = torch.cat([self.ctgVal.detach()] * 3, dim=0)
            if circularUnits=='deg':
                circularConstant = 360
            elif circularUnits=='rad':
                circularConstant = 2*np.pi
            # Add circle to last segment
            ctgVal[(self.nClasses*2):] = ctgVal[(self.nClasses*2):] + \
                circularConstant 
        elif variableType == 'linear':
            respMean = self.respMean.detach()
            respCov = self.respCov.detach()
            ctgVal = self.ctgVal.detach()
        # Interpolate the means 
        self.respMean = torch.tensor(ag.interpolate_means(
            respMean=respMean, nPoints=nPoints, method=method))
        # Interpolate the covariances
        self.respCov = torch.tensor(ag.covariance_interpolation(
            covariances=respCov, nPoints=nPoints, metric=metric, method=method))
        # Interpolate category values
        self.ctgVal = torch.tensor(ag.interpolate_means(
            respMean=ctgVal.unsqueeze(1),
            nPoints=nPoints, method='geodesic').squeeze())
        # If variable is circular, crop back to the middle segment
        if variableType == 'circular':
            iInd = self.nClasses*(nPoints+1)
            fInd = iInd + self.nClasses * (nPoints+1)
            self.respMean = self.respMean[iInd:fInd,:]
            self.respCov = self.respCov[iInd:fInd,:,:]
            self.ctgVal = self.ctgVal[iInd:fInd]


    #########################
    ### FUNCTIONS FOR GETTING MODEL OUTPUTS FOR INPUT STIMULI
    #########################

    def get_responses(self, s, addStimNoise=True, addRespNoise=True):
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
        # 1) If requested, add noise
        if addStimNoise:
            s = s + self.stimNoiseGen.rsample([nStim])
        # 2) Normalize the stimuli
        s = au.normalize_stimuli_channels(s=s, nChannels=self.nChannels)
        # 3) Append fixed and trainable filters together
        fAll = self.fixed_and_trainable_filters()
        # 4) Apply filters to the stimuli
        resp = torch.einsum('fd,nd->nf', fAll, s)
        # ADD NARROWBAND NORMALIZATION
        # 5) If requested, add response noise
        if addRespNoise:
            resp = resp + self.respNoiseGen.rsample([nStim])
        return resp


    def get_log_likelihood(self, s, addStimNoise=True, addRespNoise=True):
        """ Compute the class posteriors for each stimulus in s. Note,
        stimuli are normalized to unit norm. If requested, noise is also
        added.
        -----------------
        Arguments:
        -----------------
            - s: stimulus matrix for which to compute posteriors. (nStim x nDim)
            - addStimNoise: Logical that indicates whether to add noise to
                the input stimuli s. Added noise has the characteristics stored
                in the class.
            - addRespNoise: Logical that indicates whether to add noise to the
                filter responses.
        -----------------
        Output:
        -----------------
            - logLikelihoods: Matrix with the log-likelihood function across
            classes for each stimulus. (nStim x nClasses)
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        # 1) Get filter responses
        resp = self.get_responses(s, addStimNoise=addStimNoise,
                addRespNoise=addRespNoise)
        # 2) Get log-likelihood from the responses
        logLikelihoods = self.resp_2_log_likelihood(resp)
        return logLikelihoods


    def get_posteriors(self, s, addStimNoise=True, addRespNoise=True):
        """ Compute the class posteriors for each stimulus in s. Note,
        stimuli are normalized to unit norm. If requested, noise is also
        added.
        -----------------
        Arguments:
        -----------------
            - s: stimulus matrix for which to compute posteriors. (nStim x nDim)
            - addStimNoise: Logical that indicates whether to add noise to
                the input stimuli s. Added noise has the characteristics stored
                in the class.
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
        logLikelihoods = self.get_log_likelihood(s, addStimNoise=addStimNoise,
                addRespNoise=addRespNoise)
        # 2) Get posteriors from log-likelihoods
        posteriors = self.log_likelihood_2_posterior(logLikelihoods)
        return posteriors


    def get_estimates(self, s, method4est='MAP', addStimNoise=True,
            addRespNoise=True):
        """ Compute latent variable estimates for each stimulus in s.
        -----------------
        Arguments:
        -----------------
            - s: stimulus matrix for which to compute posteriors. (nStim x nDim)
            - addStimNoise: Logical that indicates whether to add noise to
                the input stimuli s. Added noise has the characteristics stored
                in the class.
            - addRespNoise: Logical that indicates whether to add noise to the
                filter responses.
        -----------------
        Output:
        -----------------
            - estimates: Vector with the estimated latent variable for each
                stimulus. (nStim)
        """
        # 1) Compute posteriors from the stimuli
        posteriors = self.get_posteriors(s, addStimNoise=addStimNoise,
                addRespNoise=addRespNoise)
        # 2) Get estimates from the posteriors
        estimates = self.posterior_2_estimate(posteriors, method4est=method4est)
        return estimates


    def resp_2_log_likelihood(self, resp):
        """ Compute log-likelihood of each class given the filter responses.
        -----------------
        Arguments:
        -----------------
            - resp: Matrix of filter responses. (nStim x nDim)
        -----------------
        Output:
        -----------------
            - logLikelihoods: Matrix with the log-likelihood function across
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
        logLikelihoods = quadratics + llConst.repeat(nStim, 1)
        return logLikelihoods


    def log_likelihood_2_posterior(self, logLikelihoods):
        """ Convert log-likelihoods to posterior probabilities.
        -----------------
        Arguments:
        -----------------
            - logLikelihoods: Matrix with the log-likelihood function across
            classes for each stimulus. (nStim x nClasses)
        -----------------
        Output:
        -----------------
            - posteriors: Matrix with the posterior distribution across classes
            for each stimulus. (nStim x nClasses)
        """
        posteriors = F.softmax(logLikelihoods, dim=1)
        return posteriors


    def posterior_2_estimate(self, posteriors, method4est='MAP', ctgVal=None):
        """ Convert posterior probabilities to estimates of the latent variable.
        -----------------
        Arguments:
        -----------------
            - posteriors: Matrix with the posterior distribution across classes
              for each stimulus. (nStim x nClasses)
            - method4est: Method to use for estimating the latent variable.
                Options are 'MAP' (maximum a posteriori) or 'MMSE' (minimum
                mean squared error).
            - ctgVal: Vector with the values of the latent variable for each
                category. If None, the values stored in the class are used.
                Must match the number of categories in the posteriors.
        -----------------
        Output:
        -----------------
            - estimates: Vector with the estimated latent variable for each
                stimulus. (nStim)
        """
        if ctgVal is None:
            ctgVal = self.ctgVal
        # Check that ctgVal has the same number of elements as the number
        # of categories in the posteriors. The two can differ if
        # category interpolation was used.
        if len(ctgVal) != posteriors.shape[1]:
            raise ValueError('''Error: ctgVal must have the same number of
                elements as the number of categories in the posteriors.''')
        if method4est=='MAP':
            # Get maximum posteriors indices of each stim, and its value
            (a, estimateInd) = torch.max(posteriors, dim=1)
            estimates = ctgVal[estimateInd]
        elif method4est=='MMSE':
            estimates = torch.einsum('nc,c->n', posteriors, ctgVal)
        return estimates


#####################
#####################
# CHILD CLASS, ISOTROPIC
#####################
#####################

class Isotropic(AMA):
    def __init__(self, sAll, ctgInd, nFilt=2, respNoiseVar=torch.tensor(0.02),
            pixelVar=torch.tensor(0), ctgVal=None, filtNorm='broadband',
            respCovPooling='post-filter', printWarnings=False, device='cpu'):
        """
        -----------------
        Isotropic AMA class
        -----------------
        This variant of AMA assumes that the added input level noise is
        isotropic. This allows to use analytic formulas to estimate the mean
        and covariance of the noisy normalized stimuli dataset. The
        method cannot normalize the input across different channels (i.e.
        it normalizes the whole stimulus at once).
        """


        # Compute and save as attributes the quadratic-moments weights
        # that are needed to compute statistics of stimuli under isotrpic noise
        # and normalization
        print('Computing weights for quadratic moments ...')
        start = time.time()
        self.device = device
        # Convert scalar inputs to tensors, if they are not already
        if not torch.is_tensor(respNoiseVar):
            respNoiseVar = torch.tensor(respNoiseVar)
        if not torch.is_tensor(pixelVar):
            pixelVar = torch.tensor(pixelVar)
        # Compute the pixel variance
        self.pixelSigma = torch.sqrt(pixelVar).to(device)
        self.set_isotropic_params(sAll=sAll)
        end = time.time()
        print(f'Done in {end-start} seconds')
        # Convert the pixel variance (only one number needed because isotropic),
        # into a pixel covariance matrix
        self.pixelCov = torch.eye(sAll.shape[1], device=self.device) * \
            pixelVar.to(self.device)
        self.amaType = 'Isotropic'
        # Analytic formulas only work with 1 channel
        self.nChannels = 1
        # Initialize parent class, which will fill in the rest of the statistics
        super().__init__(sAll=sAll, ctgInd=ctgInd, nFilt=nFilt,
                respNoiseVar=respNoiseVar, ctgVal=ctgVal, filtNorm=filtNorm,
                respCovPooling=respCovPooling, printWarnings=printWarnings,
                device=device)


    def set_isotropic_params(self, sAll):
        """ If stimulus noise is isotropic, set some parameters
        related to stimulus noise, and precompute quantities that will
        save compute time.
        -----------------
        Arguments:
        -----------------
            - sAll: Stimulus dataset used to compute statistics
            - pixelSigma: Standard deviation of input noise
        """
        # Weigths for the second moments
        nc, meanW, noiseW = qm.compute_isotropic_formula_weights(s=sAll,
                sigma=self.pixelSigma)
        # Square mean of stimuli divided by standard deviation
        self.nonCentrality = nc.to(self.device)
        # Weighting factors for the mean outer product in second moment estimation
        self.smMeanW = meanW.to(self.device)
        # Weighting factors for the identity matrix in second moment estimation
        self.smNoiseW = noiseW.to(self.device)
        # Expected value of the inverse of the norm of each noisy stimulus
        self.invNormMean = qm.inv_ncx_batch(mu=sAll,
                                            sigma=self.pixelSigma).to(self.device)


    def compute_norm_stim_mean(self, s, ctgInd, sameAsInit=True):
        """ Compute the mean of the noisy normalized stimuli across the
        dataset. Uses some of the noise model properties
        stored as attributes of the class.
        -----------------
        Arguments:
        -----------------
            - s: Input stimuli. (nStim x nDim)
            - ctgInd: Category index of each stimulus. (nStim)
        -----------------
        Outputs:
        -----------------
            - stimMean: Mean of the noisy normalized stimuli for each category.
                (nClasses x nDim)
        """
        print('Computing normalized noisy stimuli means ...')
        start = time.time()
        # If it's the same stimuli as initialization, use precomputed qm params
        if sameAsInit:
            invNormMean = self.invNormMean
        else:
            invNormMean = None
        # Compute mean with isotrpic formula
        stimMean = qm.isotropic_mean_batch(s=s, sigma=self.pixelSigma,
                invNormMean=invNormMean, ctgInd=ctgInd)
        end = time.time()
        print(f'Done in {end-start} seconds')
        return stimMean


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
        print('Computing normalized noisy stimuli covariances ...')
        start = time.time()
        if sameAsInit:
            # If same stimuli as initialization, use stored
            # stim mean, and precomputed qm weigths
            stimSecondM = qm.isotropic_ctg_secondM(s=s,
                    sigma=self.pixelSigma, ctgInd=ctgInd,
                    noiseW=self.smNoiseW, meanW=self.smMeanW)
            stimCov = qm.secondM_2_cov(secondM=stimSecondM,
                    mean=self.stimMean)
        else:
            # If not same stimuli as initialization, compute their
            # mean, and don't use precomputed qm weigths
            stimMean = qm.isotropic_mean_batch(s=s,
                    sigma=self.pixelSigma, ctgInd=ctgInd)
            stimSecondM = qm.isotropic_ctg_secondM(s=s,
                    sigma=self.pixelSigma, ctgInd=ctgInd)
            stimCov = qm.secondM_2_cov(secondM=stimSecondM,
                    mean=stimMean)
        end = time.time()
        print(f'Done in {end-start} seconds')
        return stimCov


    def compute_response_mean(self, s=None, ctgInd=None, sAmp=None, sameAsInit=True):
        """ Compute the mean of the filter responses to the noisy stimuli for each class.
        Note that this are the noiseless filters.
        -----------------
        Arguments:
        -----------------
            - s: stimulus matrix for which to compute the mean responses.
                If normalization is broadband and sameAsInit=True, it is
                not required. (nStim x nDim)
            - ctgInd: Vector with index categories for the stimuli in i. (length nStim)
            - sAmp: Optional to save compute when normalization='narrowband'.
                Amplitude spectrum of NORMALIZED stimuli. Should be computed with
                au.compute_amplitude_spectrum(s)' (nStim x nDim)
            - sameAsInit: Logical indicating whether it is the same stimulus set
                as for initialization. Indicates whether to use precomputed parameters.
        -----------------
        Outputs:
        -----------------
            - respMean: Mean responses of each model filter to the noisy normalized
            stimuli of each class. (nClasses x nFilt)
        """
        fAll = self.fixed_and_trainable_filters()
        if self.filtNorm=='broadband':
            if sameAsInit:
                respMean = torch.einsum('cd,kd->ck', self.stimMean, fAll)
            else:
                if s is None:
                    raise ValueError('''Error: If not same stimuli as initialization,
                            you need to provide new stimuli as input''')
                respMean = qm.isotropic_ctg_resp_mean(s=s, sigma=self.pixelSigma,
                        f=fAll, normalization=self.filtNorm, ctgInd=ctgInd)
        elif self.filtNorm=='narrowband':
            if sameAsInit:
                invNormMean = self.invNormMean
            else:
                invNormMean = None
            respMean = qm.isotropic_ctg_resp_mean(s=s, sigma=self.pixelSigma, f=fAll,
                    normalization=self.filtNorm, ctgInd=ctgInd, sAmp=sAmp,
                    invNormMean=invNormMean)
        return respMean


    def compute_response_cov(self, s=None, ctgInd=None, sAmp=None, sameAsInit=True):
        """ Compute the mean of the filter responses to the noisy stimuli for each class.
        Note that this are the noiseless filters.
        -----------------
        Arguments:
        -----------------
            - s: stimulus matrix for which to compute the mean responses.
                If normalization is broadband and sameAsInit=True, it is
                not required. (nStim x nDim)
            - ctgInd: Vector with index categories for the stimuli in i. (length nStim)
            - sAmp: Optional to save compute when normalization='narrowband'.
                Amplitude spectrum of NORMALIZED stimuli. Should be computed with
                au.compute_amplitude_spectrum(s)' (nStim x nDim)
            - sameAsInit: Logical indicating whether it is the same stimulus set
                as for initialization. Indicates whether to use precomputed parameters.
        -----------------
        Outputs:
        -----------------
            - respCov: Covariance of filter responses to the noisy normalized
            stimuli of each class. (nClasses x nFilt x nFilt)
        """
        fAll = self.fixed_and_trainable_filters()
        ### Simplest method, only works for broadband normalization
        if self.respCovPooling=='pre-filter':
            if self.filtNorm=='narrowband':
                raise ValueError('''Error: pre-filter pooling can only
                take broadband normalization''')
            if sameAsInit:
                # If sameAsInit, use precomputed stim covariances
                respCov = torch.einsum('kd,cdb,mb->ckm', fAll, self.stimCov, fAll)
            else:
                if s is None:
                    raise ValueError('''Error: If not same stimuli as initialization,
                            you need to provide new stimuli as input''')
                # If not sameAsInit, use new stim to compute covariances
                stimCov = self.compute_norm_stim_cov(s=s, ctgInd=ctgInd,
                        sameAsInit=sameAsInit)
                respCov = torch.einsum('kd,cdb,mb->ckm', fAll, stimCov, fAll)
        ### More complex, but more accurate method
        elif self.respCovPooling=='post-filter':
            # Get the parameters needed to weight different stimuli
            if sameAsInit:
                # If these are the initialization stimuli, use precomputed parameters
                meanW = self.smMeanW
                noiseW = self.smNoiseW
            else:
                nc, meanW, noiseW = qm.compute_isotropic_formula_weights(s=s,
                        sigma=self.pixelSigma)
                meanW = meanW.to(s.device)
                noiseW = noiseW.to(s.device)
            if self.filtNorm=='narrowband':
                # If narrowband, get the parameters needed to implement normalization
                if sAmp is not None:
                    # Compute similarity scores
                    similarities = qm.compute_amplitude_similarity(s=sAmp, f=fAll,
                            stimSpace='fourier', filterSpace='signal')
                    normFactors = 1/similarities
                else:
                    normFactors = None
            ### Compute the second moment matrix
            respSM = qm.isotropic_ctg_resp_secondM(s=s, f=fAll, sigma=self.pixelSigma,
                    noiseW=noiseW, meanW=meanW, normalization=self.filtNorm,
                    ctgInd=ctgInd, normFactors=normFactors)
            # Convert second moment matrix to covariance matrix using response means
            if sameAsInit and self.printWarnings:
                print('''Warning: Response covariance updating is assuming
                        response means are already updated''')
                respCov = qm.secondM_2_cov(secondM=respSM, mean=self.respMean)
            else:
                respMean = self.compute_response_mean(s=s, ctgInd=ctgInd,
                        sAmp=sAmp, sameAsInit=sameAsInit)
                respCov = qm.secondM_2_cov(secondM=respSM, mean=respMean)
        respCov = (respCov + respCov.transpose(1,2))/2
        return respCov


#    def compute_normalized_stimulus_amplitude_spec(self, s, sameAsInit=True):
#        """ Compute the amplitude spectrum of the mean normalized noisy
#        stimuli s. It uses the approximation
#        FFT(E(s/||s||))) ~ FFT(s*E(1/||s||)), like in other parts of the
#        code.
#        #
#        Arguments:
#            - s: Stimuli matrix. (nStim x nDim)
#            - sameAsInit: Logical indicating whether it is the same stimulus
#            set as for initialization. Indicates whether to use
#            precomputed parameters.
#        Outputs:
#            - sAmp: Amplitude spectra of the expected noisy normalized stimuli.
#        """
#        if sameAsInit:
#            invNormMean = self.invNormMean
#        else:
#            invNormMean = qm.inv_ncx_batch(mu=s, sigma=self.pixelSigma)
#        normalizedStim = torch.multiply(s, invNormMean.unsqueeze(1))
#        sAmp = au.compute_amplitude_spectrum(s=normalizedStim)
#        return sAmp


#####################
#####################
# CHILD CLASS, EMPIRICAL
#####################
#####################


class Empirical(AMA):
    def __init__(self, sAll, ctgInd, nFilt=2, respNoiseVar=torch.tensor(0.02),
            pixelCov=torch.tensor(0), ctgVal=None, filtNorm='broadband',
            respCovPooling='post-filter', samplesPerStim=5, nChannels=1,
            printWarnings=False, device='cpu'):
        """
        -----------------
        Empirical AMA class
        -----------------
        This variant of AMA uses empirical estimates of the noisy normalized
        stimuli means and covariances. It can normalize the input across
        different channels (i.e. left and right eye can be normalized
        separately).
        """
        # Set device
        self.device = device
        # Turn noise parameters into tensors in device, and if needed convert
        # scalar into matrix
        if not torch.is_tensor(respNoiseVar):
            respNoiseVar = torch.tensor(respNoiseVar)
        if not torch.is_tensor(pixelCov):
            pixelCov = torch.tensor(pixelCov)
        pixelCov = pixelCov.to(self.device)
        if pixelCov.dim()==0:
            self.pixelCov = torch.eye(sAll.shape[1], device=self.device) * pixelCov
        else:
            if pixelCov.shape[0] != sAll.shape[1]:
                raise ValueError('''Error: Stimulus noise covariance needs to
                        have the same dimensions as the stimuli''')
        # Set the number of channels to normalize separately
        self.nChannels = nChannels
        # Generate noisy stimuli samples to use for initialization
        n, d = sAll.shape
        noise = MultivariateNormal(torch.zeros(d, device=self.device), self.pixelCov)
        # Repeat sAll for samplesPerStim times along a new dimension
        sAllRepeated = sAll.repeat(samplesPerStim, 1, 1)
        # Generate noise samples and add them to the repeated sAll tensor
        noiseSamples = noise.sample((samplesPerStim, n))
        sAllNoisy = sAllRepeated + noiseSamples
        sAllNoisy = sAllNoisy.transpose(0, 1).reshape(-1, d)
        ctgIndNoisy = ctgInd.repeat_interleave(samplesPerStim)
        sAllNoisy = au.normalize_stimuli_channels(sAllNoisy, nChannels=nChannels)
        self.amaType = 'Empirical'
        self.device = device
        # Initialize parent class, which will fill in the rest of the statistics
        super().__init__(sAll=sAllNoisy, ctgInd=ctgIndNoisy, nFilt=nFilt,
                respNoiseVar=respNoiseVar, ctgVal=ctgVal, filtNorm=filtNorm,
                respCovPooling=respCovPooling, printWarnings=printWarnings,
                device=device)


    def compute_norm_stim_mean(self, s, ctgInd, isNormalized=True):
        """ Compute the mean of the noisy normalized stimuli across the
        dataset.
        -----------------
        Arguments:
        -----------------
            - s: Input NOISY stimuli. (nStim x nDim)
            - ctgInd: Category index of each stimulus. (nStim)
            - isNormalized: Boolean indicating if the input stimuli s
                have already been normalized.
        -----------------
        Outputs:
        -----------------
            - stimMean: Mean of the noisy normalized stimuli for each category.
                (nClasses x nDim)
        """
        print('Computing normalized noisy stimuli means ...')
        start = time.time()
        if not isNormalized:
            s = au.normalize_stimuli_channels(s=s, nChannels=self.nChannels)
        # Compute the mean of noisy normalized stimuli for each category
        stimMean = au.category_means(s=s, ctgInd=ctgInd)
        end = time.time()
        print(f'Done in {end-start} seconds')
        return stimMean


    def compute_norm_stim_cov(self, s, ctgInd, isNormalized=True):
        """ Compute the covariance across the stimulus dataset for the noisy
        normalized stimuli. Uses some of the noise model properties
        stored as attributes of the class.
        -----------------
        Arguments:
        -----------------
            - s: Input NOISY stimuli. (nStim x nDim)
            - ctgInd: Category index of each stimulus. (nStim)
            - isNormalized: Boolean indicating if the input stimuli s
                have already been normalized.
        -----------------
        Outputs:
        -----------------
            - stimCov: Mean of the noisy normalized stimuli for each category.
                (nClasses x nDim)
        """
        print('Computing normalized noisy stimuli covariances ...')
        start = time.time()
        if not isNormalized:
            s = au.normalize_stimuli_channels(s=s, nChannels=self.nChannels)
        stimSecondM = au.category_secondM(s=s, ctgInd=ctgInd)
        if self.printWarnings:
            print('''Warning: Response covariance updating is assuming
                    response means are already updated''')
        stimCov = qm.secondM_2_cov(secondM=stimSecondM, mean=self.stimMean)
        end = time.time()
        print(f'Done in {end-start} seconds')
        return stimCov


    def compute_response_mean(self, s=None, ctgInd=None, sAmp=None, isNormalized=True):
        """ Compute the mean of the filter responses to the noisy stimuli
        for each class. Note that this are the noiseless filters.
        -----------------
        Arguments:
        -----------------
            - s: NOISY stimulus matrix for which to compute the mean responses.
                If normalization is broadband it is not required. (nStim x nDim)
            - ctgInd: Vector with index categories for the stimuli in i. (length nStim)
            - sAmp: Optional to save compute when normalization='narrowband'.
                Amplitude spectrum of NOISY NORMALIZED stimuli. (nStim x nDim)
            - isNormalized: Boolean indicating if the input stimuli s
                have already been normalized.
        -----------------
        Outputs:
        -----------------
            - respMean: Mean responses of each model filter to the noisy normalized
                stimuli of each class. (nClasses x nFilt)
        """
        fAll = self.fixed_and_trainable_filters()
        if self.filtNorm=='broadband':
            respMean = torch.einsum('cd,kd->ck', self.stimMean, fAll)
        elif self.filtNorm=='narrowband':
            if not isNormalized:
                s = au.normalize_stimuli_channels(s=s, nChannels=self.nChannels)
            if sAmp is None:
                sAmp = au.compute_amplitude_spectrum(s=s)
            similarities = qm.compute_amplitude_similarity(s=sAmp, f=fAll,
                    stimSpace='fourier', filterSpace='signal')
            normFactors = 1/similarities
            responses = torch.einsum('nd,kd,n->nk', s, fAll, normFactors)
            respMean = au.category_means(s=responses, ctgInd=ctgInd)
        return respMean


    def compute_response_cov(self, s=None, ctgInd=None, sAmp=None, isNormalized=True):
        """ Compute the mean of the filter responses to the noisy stimuli for each class.
        Note that this are the noiseless filters.
        -----------------
        Arguments:
        -----------------
            - s: NOISY stimulus matrix for which to compute the mean responses.
                If normalization is broadband and sameAsInit=True, it is
                not required. (nStim x nDim)
            - ctgInd: Vector with index categories for the stimuli in i. (length nStim)
            - sAmp: Optional to save compute when normalization='narrowband'.
                Amplitude spectrum of NORMALIZED stimuli. Should be computed with
                'au.compute_amplitude_spectrum(s)' (nStim x nDim)
            - isNormalized: Boolean indicating if the input stimuli s
                have already been normalized.
        -----------------
        Outputs:
        -----------------
            - respCov: Covariance of filter responses to the noisy normalized
                stimuli of each class. (nClasses x nFilt x nFilt)
        """
        fAll = self.fixed_and_trainable_filters()
        ### Simplest method, only works for broadband normalization
        if self.respCovPooling=='pre-filter':
            if self.filtNorm=='narrowband':
                raise ValueError('''Error: pre-filter pooling can only
                take broadband normalization''')
            respCov = torch.einsum('kd,cdb,mb->ckm', fAll, self.stimCov, fAll)
        ### More complex, but more accurate method
        elif self.respCovPooling=='post-filter':
            if not isNormalized:
                s = au.normalize_stimuli_channels(s=s, nChannels=self.nChannels)
            # Get the filter responses
            if self.filtNorm=='narrowband':
                if sAmp is not None:
                    sAmp = au.compute_amplitude_spectrum(s=s)
                similarities = qm.compute_amplitude_similarity(s=sAmp, f=fAll,
                        stimSpace='fourier', filterSpace='signal')
                normFactors = 1/similarities
                responses = torch.einsum('nd,kd,n->nk', s, fAll, normFactors)
            elif self.filtNorm=='broadband':
                responses = torch.einsum('nd,kd,n->nk', s, fAll)
            # Compute responses SM
            ### Compute the second moment matrix
            respSM = au.category_secondM(s=responses, ctgInd=ctgInd)
            if self.printWarnings:
                print('''Warning: Response covariance updating is assuming
                        response means were already updated''')
            respCov = qm.secondM_2_cov(secondM=respSM, mean=self.respMean)
        respCov = (respCov + respCov.transpose(1,2))/2
        return respCov


