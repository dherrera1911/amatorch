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
import time

#####################
#####################
# PARENT AMA CLASS
#####################
#####################

class AMA(ABC, nn.Module):
    def __init__(self, sAll, ctgInd, nFilt=2, respNoiseVar=torch.tensor(0.02),
            ctgVal=None, filtNorm='broadband', respCovPooling='pre-filter'):
        """ AMA model object.
        Arguments:
        sAll: Input stimuli. (nStim x nDim)
        ctgInd: Category index of each stimulus. (nStim)
        nFilt: Number of filters to train
        respNoiseVar: Variance of filter response noise. Scalar
        ctgVal: Value of the latent variable corresponding to each category.
            Defaults to equispaced points in [-1, 1].
        filtNorm: String indicating method use to normalize filter responses.
            Can be 'broadband' (default) or 'narrowband'
        respCovPooling: String indicating how to compute the filter response
            covariances. 'pre-filter' can be less accurate depending on the
            problem, but is faster. 'post-filter' is slower but gives
            exact results. Defults to 'post-filter'
        """
        super().__init__()
        print('Initializing AMA')
        ### Make initial random filters
        fInit = torch.randn(nFilt, sAll.shape[1])
        fInit = F.normalize(fInit, p=2, dim=1)
        # Model parameters
        self.f = nn.Parameter(fInit)
        geotorch.sphere(self, "f")
        # Attribute with fixed (non-trainable) filters
        self.fFixed = torch.tensor([])
        # Assign handy variables
        self.nFilt = self.f.shape[0]  # Number of trainable filters
        self.nFiltAll = self.nFilt  # Number of filters including fixed filters
        self.nDim = self.f.shape[1]  # Number of dimensions
        self.nClasses = np.unique(ctgInd).size  # Number of classes
        self.filtNorm = filtNorm  # Method to normalize the filters
        self.respCovPooling = respCovPooling  # Method to generate response covariance
        # If no category values given, assign equispaced values in [-1,1]
        if ctgVal == None:
            ctgVal = np.linspace(-1, 1, self.nClasses)
        self.ctgVal = ctgVal
        # Make filter noise matrix (##IMPLEMENT GENERAL COVARIANCE LATER##)
        self.respNoiseVar = torch.tensor(respNoiseVar)
        self.respNoiseCov = torch.eye(self.nFiltAll) * self.respNoiseVar
        # Make random number generators for pixel and response noise
        self.stimNoiseGen = MultivariateNormal(loc=torch.zeros(self.nDim),
                covariance_matrix=self.pixelCov)
        self.respNoiseGen = MultivariateNormal(loc=torch.zeros(self.nFiltAll),
                covariance_matrix=self.respNoiseCov)
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
        Output:
        - fAll: Tensor with all filters. (nFilt x nDim)
        """
        return torch.cat((self.fFixed, self.f))


    #########################
    ### FUNCTIONS FOR UPDATING MODEL STATISTICS AND FILTERS
    #########################

    # Methods for updating model statistics depend on the procedure to
    # be used, and on the model's architecture. Define abstract methods
    # that are implemented in the child classes.

    @abstractmethod
    def compute_norm_stim_mean(self, s, ctgInd):
        pass

#    @abstractmethod
#    def compute_norm_stim_mean_individual(self, sAll):
#        pass

    @abstractmethod
    def compute_norm_stim_cov(self, s, ctgInd):
        pass

    @abstractmethod
    def compute_response_cov(self, s, ctgInd):
        pass


#    @abstractmethod
#    def compute_norm_stim_cov_individual(self, sAll):
#        pass

    @abstractmethod
    def update_response_statistics(self):
        pass


    def compute_response_mean(self, s=None, ctgInd=None):
        ### Add sAmp
        fAll = self.fixed_and_trainable_filters()
        if self.filtNorm == 'broadband':
            respMean = torch.einsum('cd,kd->ck', self.stimMean, fAll)
        else:
            print('Error: Compute response mean only implemented for simplest case')
            # To implement this, use the child-specific functions that return the
            # individual mean of each stimulus. Then use the individual means,
            # together with narrowband weighting to compute the
            # response means
        return respMean


    def update_response_statistics(self, sAll, ctgInd, sAmp=None,
            sameAsInit=True):
        """ Update (in place) the conditional response means and covariances
        to match the current object filters
        Arguments:
        - sAll: Stimulus dataset to use for response updating. Ideally, it is
         the same dataset used to initialize the model, for which there are
         precomputed quantities. (nStim x nDim)
        - ctgInd: Vector with category indices of the stimuli
        - sAmp: Optional to save compute. Pre-computed amplitude spectrum
        of the stimulus dataset. Should be computed
        with 'qm.compute_amplitude_spectrum'. (nStim x nDim)
        - sameAsInit: Logical indicating whether sAll is the same dataset
        used for initialization. In that case, precomputed values are used
        to get the statistics
        """
        # Get all filters (fixed and trainable)
        fAll = self.fixed_and_trainable_filters()
        self.nFilt = self.f.shape[0]
        self.nFiltAll = fAll.shape[0]
        # If new filters were manually added, expand the response noise covariance
        self.respNoiseCov = torch.eye(self.nFiltAll) * self.respNoiseVar
        # Update covariances, size nClasses*nFilt*nFilt
        # Assign precomputed valeus, if same as initialization
        if self.filtNorm == 'broadband':
            self.respMean = self.compute_response_mean(s=sAll,
                    ctgInd=ctgInd)
            self.respCovNoiseless = self.compute_response_cov(s=sAll,
                    ctgInd=ctgInd)
        elif self.filtNorm == 'narrowband':
            sAmp = compute_amplitude_spectrum(s)
            print('Need to implement narrowband or post filter')
        # Add response noise to the stimulus-induced variability of responses
        self.respCov = self.respCovNoiseless + \
            self.respNoiseCov.repeat(self.nClasses, 1, 1)
        self.respNoiseGen = MultivariateNormal(loc=torch.zeros(self.nFiltAll),
                covariance_matrix=self.respNoiseCov)


    #### Consider making this child-specific
    def assign_filter_values(self, fNew, sAll, ctgInd, sAmp=None,
            sameAsInit=True):
        """ Overwrite the values to the model filters. Updates model
        parameters and statistics accordingly.
        Arguments:
        - fNew: Matrix with the new filters as rows. The new number of filters
            doesn't need to match the old number. (nFilt x nDim)
        - Rest of inputs, sAll, ctgInd, sAmp, sameAsInit, are as explained
        in update_response_statistics()
        """
        # Remove parametrization so we can change the filters
        parametrize.remove_parametrizations(self, "f", leave_parametrized=True)
        # Model parameters. Important to clone fNew, otherwise geotorch
        # modifies the original
        self.f = nn.Parameter(fNew.clone())
        geotorch.sphere(self, "f")
        self.f = fNew
        # Update model values
        self.update_response_statistics(sAll=sAll, ctgInd=ctgInd, sAmp=sAmp,
                sameAsInit=sameAsInit)
        # In case new filters were added, adapt the noise generator
        self.respNoiseGen = MultivariateNormal(loc=torch.zeros(self.nFiltAll),
                covariance_matrix=self.respNoiseCov)


    def reinitialize_trainable(self, sAll, ctgInd, sAmp=None, sameAsInit=True):
        """ Re-initialize the trainable filters to random values.
        Input parameters are as in update_response_statistics()
        """
        fRandom = torch.randn(self.nFilt, self.nDim)
        fRandom = F.normalize(fRandom, p=2, dim=1)
        # Statistics are updated in assign_filter_values
        self.assign_filter_values(fRandom, sAll=sAll, ctgInd=ctgInd,
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
        Arguments:
        - fFixed: Te tensor with the new filters. (nFilt x nDim)
        - Rest of input parameters are as in update_response_statistics()
        """
        self.fFixed = fFixed.clone()
        self.update_response_statistics(sAll=sAll, ctgInd=ctgInd, sAmp=None,
                sameAsInit=True)


    def add_new_filters(self, nFiltNew, sAll, ctgInd, sAmp=None,
            sameAsInit=True):
        """ Add new, random filters to the filters already contained in
        the model. Adapt model statistics and parameters accordingly.
        Arguments:
        - nFiltNew: number of new fiters to add
        - Rest of inputs, sAll, ctgInd, sAmp, sameAsInit, are as explained
        in update_response_statistics()
        """
        # Initialize new random filters and set length to 1 
        fNew = F.normalize(torch.randn(nFiltNew, self.nDim), p=2, dim=1)
        fOld = self.f.detach().clone()
        f = torch.cat((fOld, fNew))  # Concatenate old and new filters
        # assign_filter_values updates statistics
        self.assign_filter_values(fNew=f, sAll=sAll, ctgInd=ctgInd, sAmp=sAmp,
                sameAsInit=sameAsInit)


    #########################
    ### FUNCTIONS FOR GETTING MODEL OUTPUTS FOR INPUT STIMULI
    #########################

    def get_responses(self, s, addStimNoise=True, addRespNoise=True):
        """ Compute the responses of the filters to each stimulus in s. Note,
        stimuli are normalized to unit norm. If requested, noise is also
        added.
        Arguments:
        - s: stimulus matrix for which to compute posteriors. (nStim x nDim)
        - addStimNoise: Logical that indicates whether to add noise to the input
            stimuli s. Added noise has the characteristics stored in the class.
        Output:
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
        s = F.normalize(s, p=2, dim=1)
        # 3) Append fixed and trainable filters together
        fAll = self.fixed_and_trainable_filters()
        # 4) Apply filters to the stimuli
        resp = torch.einsum('fd,nd->nf', fAll, s)
        # 2) If requested, add response noise
        if addRespNoise:
            resp = resp + self.respNoiseGen.rsample([nStim])
        return resp


    def get_log_likelihood(self, s, addStimNoise=True, addRespNoise=True):
        """ Compute the class posteriors for each stimulus in s. Note,
        stimuli are normalized to unit norm. If requested, noise is also
        added.
        Arguments:
        - s: stimulus matrix for which to compute posteriors. (nStim x nDim)
        - addStimNoise: Logical that indicates whether to add noise to
            the input stimuli s. Added noise has the characteristics stored
            in the class.
        - addRespNoise: Logical that indicates whether to add noise to the
            filter responses.
        Output:
        - log_likelihood: Matrix with the log-likelihood function across
        classes for each stimulus. (nStim x nClasses)
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        nStim = s.shape[0]
        # 1) Get filter responses
        resp = self.get_responses(s, addStimNoise=addStimNoise,
                addRespNoise=addRespNoise)
        # 2) Difference between responses and class means. (nStim x nClasses x nFilt)
        respDiff = resp.unsqueeze(1).repeat(1, self.nClasses, 1) - \
                self.respMean.unsqueeze(0).repeat(nStim, 1, 1)
        ## Get the log-likelihood of each class
        # 3) Quadratic component of log-likelihood (with negative sign)
        quadratics = -0.5 * torch.einsum('ncd,cdb,ncb->nc', respDiff,
                self.respCov.inverse(), respDiff)
        # 4) Constant term of log-likelihood
        llConst = -0.5 * self.nFiltAll * torch.log(2*torch.tensor(torch.pi)) - \
            0.5 * torch.logdet(self.respCov)
        # 5) Add quadratics and constants to get log-likelihood
        log_likelihood = quadratics + llConst.repeat(nStim, 1)
        return log_likelihood


    def get_posteriors(self, s, addStimNoise=True, addRespNoise=True):
        """ Compute the class posteriors for each stimulus in s. Note,
        stimuli are normalized to unit norm. If requested, noise is also
        added.
        Arguments:
        - s: stimulus matrix for which to compute posteriors. (nStim x nDim)
        - addStimNoise: Logical that indicates whether to add noise to
            the input stimuli s. Added noise has the characteristics stored
            in the class.
        - addRespNoise: Logical that indicates whether to add noise to the
            filter responses.
        Output:
        - posteriors: Matrix with the posterior distribution across classes
        for each stimulus. (nStim x nClasses)
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        # 1) Get log-likelihoods
        log_likelihoods = self.get_log_likelihood(s, addStimNoise=addStimNoise,
                addRespNoise=addRespNoise)
        # 2) Add quadratics and constants and softmax to get posterior probs
        posteriors = F.softmax(log_likelihoods, dim=1)
        return posteriors


    def get_estimates(self, s, method4est='MAP', addStimNoise=True,
            addRespNoise=True):
        """ Compute latent variable estimates for each stimulus in s.
        #
        Arguments:
        - s (nStim x nDim) is stimulus matrix
        #
        Output:
        - estimates (nStim). Vector with the estimate for each stimulus """
        posteriors = self.get_posteriors(s, addStimNoise=addStimNoise,
                addRespNoise=addRespNoise)
        if method4est=='MAP':
            # Get maximum posteriors indices of each stim, and its value
            (a, estimateInd) = torch.max(posteriors, dim=1)
            estimates = self.ctgVal[estimateInd]
        elif method4est=='MMSE':
            estimates = torch.einsum('nc,c->n', posteriors, self.ctgVal)
        return estimates


    def compute_stimulus_filter_similarity(self, s, sAmp=None):
        """ Compute the similarity between each stimulus and the model
        filters, according to the similarity type indicated in model
        initialization.
        #
        Arguments:
        - s: Stimulus matrix. (nStim x nDim)
        - sAmp: Optional to save compute. Pre-computed amplitude spectrum
        of the stimulus dataset. Should be computed with
        'qm.compute_amplitude_spectrum'. (nStim x nDim)
        #
        Output:
        - similarity: Similarity matrix between each stimulus and
        each filter. (nStim x nFilt)"""
        ### TO IMPLEMENT:
        # Implement signal-space similarity computing
        if sAmp is None:
            sAmp = qm.compute_amplitude_spectrum(s)
        fAll = self.fixed_and_trainable_filters()
        fAmp = qm.compute_amplitude_spectrum(fAll)
        # Get the normalization factor, i.e. inverse product of norms
        normFactor = torch.einsum('n,k->nk', 1/sAmp.norm(dim=1),
                1/fAmp.norm(dim=1))
        # Compute the Amplitude spectrum similarity
        similarity = torch.einsum('nd,kd,nk->nk', sAmp, fAmp, normFactor)
        return similarity


#####################
#####################
# CHILD CLASS, ISOTROPIC
#####################
#####################

class Isotropic(AMA):
    def __init__(self, sAll, ctgInd, nFilt=2, respNoiseVar=torch.tensor(0.02),
            ctgVal=None, filtNorm='broadband', respCovPooling='post-filter',
            pixelVar=torch.tensor(0)):
        # Compute and save as attributes the quadratic-moments weights
        # that are needed to compute statistics of stimuli under isotrpic noise
        # and normalization
        print('Computing weights for quadratic moments ...')
        start = time.time()
        self.set_isotropic_params(sAll=sAll,
                pixelSigma=torch.sqrt(torch.tensor(pixelVar)))
        end = time.time()
        print(f'Done in {end-start} seconds')
        # Convert the pixel variance (only one number needed because isotropic),
        # into a pixel covariance matrix
        self.pixelCov = torch.eye(sAll.shape[1]) * torch.tensor(pixelVar)
        # Initialize parent class, which will fill in the rest of the statistics
        super().__init__(sAll=sAll, ctgInd=ctgInd, nFilt=nFilt,
                respNoiseVar=respNoiseVar, ctgVal=ctgVal, filtNorm=filtNorm,
                respCovPooling=respCovPooling)


    def set_isotropic_params(self, sAll, pixelSigma):
        """ If stimulus noise is isotropic, set some parameters
        related to stimulus noise, and precompute quantities that will
        save compute time.
        Arguments:
        - sAll: Stimulus dataset used to compute statistics
        - pixelSigma: Standard deviation of input noise
        """
        nDim = sAll.shape[1]
        self.pixelSigma = pixelSigma
        # Weigths for the second moments
        nc, meanW, noiseW = qm.compute_isotropic_formula_weights(s=sAll,
                sigma=pixelSigma)
        # Square mean of stimuli divided by standard deviation
        self.nonCentrality = nc
        # Weighting factors for the mean outer product in second moment estimation
        self.smMeanW = meanW
        # Weighting factors for the identity matrix in second moment estimation
        self.smNoiseW = noiseW
        # Expected value of the inverse of the norm of each noisy stimulus
        self.invNormMean = qm.inv_ncx_batch(mu=sAll, sigma=pixelSigma)


    def compute_norm_stim_mean(self, s, ctgInd, sameAsInit=True):
        """ Compute the mean of the noisy normalized stimuli across the
        dataset. Uses some of the noise model properties
        stored as attributes of the class.
        """
        print('Computing normalized noisy stimuli means ...')
        start = time.time()
        # If it's the same stimuli as initialization, use precomputed qm params
        if sameAsInit:
            invNormMean = self.invNormMean
        else:
            invNormMean = None
        # Compute mean with isotrpic formula
        stimMean = qm.isotropic_weighted_mean_batch(s=s,
                sigma=self.pixelSigma, invNormMean=invNormMean,
                ctgInd=ctgInd)
        end = time.time()
        print(f'Done in {end-start} seconds')
        return stimMean


#    def compute_norm_stim_mean_individual(self, s, ctgInd, sameAsInit=True):
#        """ Compute the mean of each noisy normalized stimulus in
#        the dataset (as opposed to the mean of the dataset). Uses some
#        of the noise model properties stored as attributes of the class.
#        """
#        print('Computing individual normalized noisy stimuli means ...')
#        start = time.time()
#        # If it's the same stimuli as initialization, use precomputed qm params
#        if sameAsInit:
#            invNormMean = self.invNormMean
#        else:
#            invNormMean = None
#        # Compute mean with isotrpic formula
#        stimMean = qm.isotropic_weighted_mean_batch(s=s,
#                sigma=self.pixelSigma, invNormMean=invNormMean,
#                ctgInd=ctgInd)
#        end = time.time()
#        print(f'Done in {end-start} seconds')
#        return stimMean


    def compute_norm_stim_cov(self, s, ctgInd, sameAsInit=True):
        """ Compute the covariance across the stimulus dataset for the noisy
        normalized stimuli. Uses some of the noise model properties
        stored as attributes of the class.
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
            stimMean = qm.isotropic_weighted_mean_batch(s=s,
                    sigma=self.pixelSigma, ctgInd=ctgInd)
            stimSecondM = qm.isotropic_ctg_secondM(s=s,
                    sigma=self.pixelSigma, ctgInd=ctgInd)
            stimCov = qm.secondM_2_cov(secondM=stimSecondM,
                    mean=stimMean)
        end = time.time()
        print(f'Done in {end-start} seconds')
        return stimCov


    def compute_response_cov(self, s=None, ctgInd=None, sAmp=None,
            sameAsInit=True):
        ### Add sAmp
        fAll = self.fixed_and_trainable_filters()
        if self.respCovPooling=='pre-filter':
            ## pre-filter pooling only accepts broadband normalization
            if self.filtNorm=='narrowband':
                raise ValueError('''Error: pre-filter pooling can only
                take broadband normalization''')
            respCov = torch.einsum('kd,cdb,mb->ckm', fAll, self.stimCov, fAll)
        else if self.respCovPooling=='post-filter':
#            # If not same as init, compute weightings of isotropic formula
#            if not sameAsInit:
#                nc, meanW, noiseW = qm.compute_isotropic_formula_weights(s=s,
#                        sigma=self.pixelSigma)
#            else if sameAsInit:
#                nc = self.nonCentrality
#                meanW = self.smMeanW
#                noiseW = self.smNoiseW
#            # First filter the stimuli to get the responses
#            responses = torch.einsum('kd,nd->nk', fAll, s)
#            if self.filtNorm=='broadband':
                raise ValueError('''Error: Response covariances only
                        implemented for simplest case for now''')
            # To implement this, use the child-specific functions that return the
            # individual mean of each stimulus. Then use the individual means,
            # together with narrowband weighting (or not) to compute the
            # response means
        return respCov


