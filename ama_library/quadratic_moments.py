import numpy as np
import torch
from torch.special import gammaln
from torch import fft as fft
import mpmath as mpm  # Important to use mpmath for hyp1f1, scipy blows up
from ama_library import utilities as au

##################################
##################################
#
## QUADRATIC MOMENTS FUNCTIONS
#
##################################
##################################
#
# This group of functions provides several functionalities
# to analytically calculate first and second moments of different
# distributions. Mainly, utilities for ratios of quadratic forms,
# inverse non-central chi squared distribution, and inverse
# non-central chi distribution are provided.

def compute_nc_parameter_batch(s, sigma):
    """ Compute the non-centrality parameter of each row
    in s, given isotropic noise with sigma standard deviation.
    ----------------
    Arguments:
    ----------------
      - s: Stimuli dataset. (nStim x nDim)
      - sigma: standard deviation of noise. Sigma can be a scalar,
          or a vector where each element of s will be divided by
          the corresponding element (it won't be an exact non-centrality
          parameter in this case)
    ----------------
    Outputs:
    ----------------
      - nc: Non-centrality parameter of each stimulus. ||mu||^2, where
          mu = s/sigma
    """
    sNorm = s/sigma
    # non-centrality parameter, ||\mu||^2. Make numpy array for mpm package
    if sNorm.is_cuda:
        sNorm = sNorm.to('cpu')
    nc = np.array(sNorm.norm(dim=1)**2)
    return nc


# Get the values of the hypergeometric function given a and b,
# for each value of non-centrality parameter, which is sitmulus dependent
def compute_stimuli_hyp1f1(a, b, nc):
    """ For each element in nc (which is the non-centrality parameter
    given by ||s/sigma||**2 for stimulus s), compute the
    confluent hypergeometric function hyp1f1(a, b, -nc[i]/2)
    ----------------
    Arguments:
    ----------------
      - a: First parameter of hyp1f1 (usually 1 or 1/2). (Scalar)
      - b: Second parameter of hyp1f1 (usually df/2+k, k being an integer). (Scalar)
      - nc: Non-centrality parameter. Has to be numpy array (Vector length df)
    ----------------
    Outputs:
    ----------------
      - hypFun: Value of hyp1f1 for each nc. (Vector length df)
    """
    nStim = len(nc)  # Get number of dimensions
    # Calculate hypergeometric functions (not vectorized function)
    hypFun = torch.zeros(nStim)
    for i in range(nStim):
        hypFun[i] = torch.tensor(float(mpm.hyp1f1(a, b, -nc[i]/2)))
    return hypFun


def compute_isotropic_formula_weights(s, sigma):
    """ For a set of stimuli, and a given standard deviation for
    isotropic Gaussian noise, compute and the weights of the
    stimulus outer products and the identity matrix needed to get
    the second moment matrix of each stimulus.
    ----------------
    Arguments:
    ----------------
      - s: Stimulus dataset. (nStim x nDim)
      - sigma: Standard deviation of the noise
    ----------------
    Outputs:
    ----------------
      - nonCentrality: Non centrality parameter of each stimulus (nStim)
      - smMeanW: Weigths for the outer products of the means
      (i.e. the stimuli). (nStim)
      - smNoiseW: Weights for the identity matrices. (nStim)
    """
    df = s.shape[1]
    # Square mean of stimuli divided by standard deviation
    nonCentrality = compute_nc_parameter_batch(s, sigma)
    # Weighting factors for the mean outer product in second moment estimation
    smMeanW = compute_stimuli_hyp1f1(a=1, b=df/2+2,
            nc=nonCentrality) * (1/(df+2))
    # Weighting factors for the identity matrix in second moment estimation
    smNoiseW = compute_stimuli_hyp1f1(a=1, b=df/2+1,
            nc=nonCentrality) * (1/df)
    return torch.tensor(nonCentrality), smMeanW, smNoiseW


# Inverse non-centered chi expectation.
def inv_ncx(mu, sigma):
    """ Get the expected value of the inverse of the norm
    of a multivariate gaussian X with mean mu and isotropic noise
    variance sigma.
    ----------------
    Arguments:
    ----------------
      - mu: Multidimensional mean of the gaussian. (Vector length df)
      - sigma: Standard deviation of isotropic noise. (Scalar)
    ----------------
    Outputs:
    ----------------
      - expectedValue: Expected value of 1/||x|| with x~N(mu, sigma).
          (Scalar)
    """
    df = len(mu)
    # lambda parameter of non-central chi distribution, squared
    lam = np.array(torch.sum((mu/sigma)**2))
    gammaQRes = (1/np.sqrt(2)) * torch.exp(gammaln(torch.tensor((df-1)/2))
                                           - gammaln(torch.tensor(df/2)))
    hypGeomVal = torch.tensor(float(mpm.hyp1f1(1/2, df/2, -lam/2)))
    expectedValue = (gammaQRes * hypGeomVal) / sigma
    return expectedValue


# Inverse non-centered chi expectation.
def inv_ncx_batch(mu, sigma):
    """ Get the expected value of the inverse of the norm
    of a multivariate gaussian X with mean mu and isotropic noise
    standard deviation sigma, for each different value of mu.
    ----------------
    Arguments:
    ----------------
      - mu: Multidimensional mean of the gaussian. (nStim x df)
      - sigma: Standard deviation of isotropic noise. (Scalar)
    ----------------
    Outputs:
    ----------------
      - expectedValue: Expected value of 1/||x|| with x~N(mu, sigma).
          (Vector length nStim)
    """
    df = mu.shape[1]
    # lambda parameter of non-central chi distribution, squared
    lam = compute_nc_parameter_batch(s=mu, sigma=sigma)
    # Corresponding hypergeometric function values
    hypGeomVal = compute_stimuli_hyp1f1(1/2, df/2, lam)
    gammaQRes = (1/np.sqrt(2)) * torch.exp(gammaln(torch.tensor((df-1)/2))
                                           - gammaln(torch.tensor(df/2)))
    expectedValue = (gammaQRes * hypGeomVal) / sigma  # This is a torch tensor
    return expectedValue


# Inverse non-centered chi square expectation.
##### Check that description is ok, regarding non-centrality
##### parameter and what distribution is actually obtained
def inv_ncx2(df, nc):
    """ Get the expected value of the inverse of the
    squared norm of a non-centered gaussian
    distribution, with degrees of freedom df, and non-centrality
    parameter nc (||\mu||^2).
    ----------------
    Arguments:
    ----------------
      - df: degrees of freedom
      - nc: non-centrality parameter
    """
    df = float(df)
    nc = float(nc)
    gammaQRes = 0.5 * torch.exp((gammaln(torch.tensor(df)/2-1) -
                                 gammaln(torch.tensor(df)/2)))
    hypFunRes = mpm.hyp1f1(1, df/2, -nc/2)
    hypFunRes = torch.tensor(float(hypFunRes))
    expectation = gammaQRes * hypFunRes
    return expectation



##################################
##################################
#
## CALCULATE STIMULI STATISTICS UNDER NORMALIZATION
#
##################################
##################################
#
# NOTE: POINT THE SPECIFIC FUNCTIONS TO THE ACCOMPANYING DOCUMENT

#############
#### Stimulus means
#############

def isotropic_individual_stim_mean(s, sigma=0.1, invNorm=None):
    """ Compute the mean of each row in s
    (i.e. each stimulus) given isotropic noise with standard
    deviation sigma and normalization by the norm. It can either take the
    expected values of the inverse norm as inputs, or
    compute them here
    ----------------
    Arguments:
    ----------------
      - s: Stimulus dataset. (nStim x nDim)
      - sigma: Standard deviation of the noise
      - invNorm: Expected value of the inverse norm of the
          noisy stimulus. (nStim)
    ----------------
    Outputs:
    ----------------
      - sNormMeanExpected: Expected mean value of each normalized
          noisy stimulus. (nStim x nDim)
    """
    if invNorm is None:
        invNorm = inv_ncx_batch(s, sigma)
    sNormMeanExpected = torch.einsum('nd,n->nd', s, invNorm)
    return sNormMeanExpected


# Compute the expected value of noisy normalized stimulus set
# by doing a weighted combination of the stimuli given
# precomputed weighting values
# previously called isotropic_weighted_mean_batch
def isotropic_mean_batch(s, sigma, invNormMean=None, ctgInd=None):
    """ Compute the expected value of the normalized noisy stimuli
    across the dataset, given by stimuli s and isotropic noise with
    sigma standard deviation.
    The expected value is computed by weighted sum of the
    stimuli, where the weights are given by the expected value of the
    inverse norm of the noisy stimulus (E(1/||s+\gamma||)).
    The sum weights can be precomputed with the inv_ncx_batch function.
    ----------------
    Arguments:
    ----------------
      - s: matrix with stimuli. (nStim x nDim)
      - sigma: noise standard deviation. Scalar. If invNormMean is
          given as input, it is ignored.
      - invNormMean: Optional to save compute. Expected inverse norm
          for each stimulus. (nStim)
      - ctgInd: Vector with index categories for the stimuli in s
    ----------------
    Outputs:
    ----------------
      - sNormMeanExpected: Expected value of the normalized noisy stimuli
          for each class in the dataset (nClasses x nDim)
    """
    nDim = s.shape[1]
    nStim = s.shape[0]
    # Fill up vector ctgInd with 0's if not given
    if ctgInd is None:
        ctgInd = torch.zeros(nStim, device=s.device)
    # If precomputed normalization factors are not given, compute them here
    if invNormMean is None:
        invNormMean = inv_ncx_batch(mu=s, sigma=sigma).to(s.device)
    # Compute the means of the second moments for each batch of stimuli
    nClasses = torch.unique(ctgInd).size()[0]
    sNormMeanExpected = torch.zeros(nClasses, nDim, device=s.device)
    for cl in range(nClasses):
        mask = (ctgInd == cl)
        sLevel = s[mask,:]  # Stimuli of the same category
        invNormMeanLevel = invNormMean[mask]
        sNormMeanExpected[cl,:] = torch.mean(torch.einsum('nb,n->nb',
            sLevel, invNormMeanLevel), dim=0)
    return sNormMeanExpected


#############
#### Stimulus second moment matrices
#############

# Apply the isotropic covariance formula to get the covariance
# for each stimulus
def isotropic_individual_stim_secondM(s, sigma, noiseW=None, meanW=None):
    """ Compute the second moment of each row in s
    (i.e. each stimulus) given isotropic noise with standard
    deviation sigma and normalization by the norm. It can either take
    the weighting terms for the identity and the mean outer product as
    inputs, or compute them here
    ----------------
    Arguments:
    ----------------
      - s: Stimulus dataset. (nStim x nDim)
      - sigma: Standard deviation of the noise
      - noiseW: Weighting term for the identity matrix. (nStim)
      - meanW: Weighting term for the mean outer product. (nStim)
    ----------------
    Outputs:
    ----------------
      - sNormSecondMExpected: Expected second moment of each normalized
          noisy stimulus. (nStim x nDim x nDim)
    """
    if s.dim() == 1:
        s = s.unsqueeze(0)
    df = s.shape[1]
    nStim = s.shape[0]
    # If precomputed weights are not given, compute them here
    if (noiseW is None) or (meanW is None):
        nc = compute_nc_parameter_batch(s, sigma)
    if noiseW is None:
        hypFunNoise = compute_stimuli_hyp1f1(a=1, b=df/2+1, nc=nc)
        noiseW = hypFunNoise * (1/df)
        noiseW = noiseW.to(s.device)
    if meanW is None:
        hypFunMean = compute_stimuli_hyp1f1(a=1, b=df/2+2, nc=nc)
        meanW = hypFunMean * (1/(df+2))
        meanW = meanW.to(s.device)
    # Compute the second moment of each stimulus
    sScaled = s/sigma
    #outerProds = torch.einsum('nd,nb->ndb', sScaled, sScaled)
    ## Get the outer product of the normalized stimulus, and multiply by weight
    #meanTerm = torch.einsum('ndb,n->ndb', outerProds, meanW)
    # Get the outer product of the normalized stimulus, and multiply by weight
    meanTerm = torch.einsum('nd,nb,n->ndb', sScaled, sScaled, meanW)
    # Multiply the identity matrix by weighting term
    noiseTerm = torch.einsum('db,n->ndb', torch.eye(df, device=s.device), noiseW)
    # Add the two terms
    expectedCovs = meanTerm + noiseTerm
    return expectedCovs.float()


# Compute the expected second moment matrix of noisy normalized stimulus
# set by doing a weighted combination of mean and noise, given
# the precomputed weighting values. Can include the weights given by
# narrowband normalization
#def isotropic_weighted_secondM_batch(s, sigma,
def isotropic_ctg_secondM(s, sigma, noiseW=None, meanW=None, ctgInd=None,
        normFactor=None):
    """ Compute the second moment of the normalized noisy stimuli
    across the dataset, given by stimuli s and isotropic noise with
    sigma standard deviation.
    The second moment is computed by weighted sum of the outer product of
    each stimulus and the identity matrix (corresponding to isotropic noise).
    The sum weights noiseW and meanW can be precomputed and given as input
    and are otherwise computed here.
    A vector of normalizing factors (normFactor) given by the filter-specific
    normalization can be passed as inputs, to simulate narrowband
    normalization.
    ----------------
    Arguments:
    ----------------
      - s: matrix with stimuli. (nStim x nDim)
      - sigma: noise standard deviation. Scalar
      - noiseW: Weight of the noise for each row in s. Obtained
          as hyp1f1(1; nDim/2+1, -(||s/sigma||**2)/2).
          If empty, it is computed inside this function. Vector, length = nStim
      - meanW: Weight of mean outer product for each row in s. Obtained
          as hyp1f1(1; nDim/2+2, -(||s/sigma||**2)/2).
          If empty, it is computed inside this function. Vector, length = nStim
      - ctgInd: Vector with index categories for the stimuli in s
      - normFactor: normalization factor given by filter-stimulus phase
          independent similarity. Either a scalar, or a vector length=nStim.
    ----------------
    Output: 
    ----------------
      - expectedSM: Array with expected second moment matrices. (nClasses x df x df)
    """
    df = s.shape[1]
    nStim = s.shape[0]
    # Fill up vector ctgInd with 0's if not given
    if ctgInd is None:
        ctgInd = torch.zeros(nStim, device=s.device)
    # If precomputed weights are not given, compute them here
    if (noiseW is None) or (meanW is None):
        nc = compute_nc_parameter_batch(s, sigma)
    if noiseW is None:
        hypFunNoise = compute_stimuli_hyp1f1(a=1, b=df/2+1, nc=nc)
        noiseW = hypFunNoise * (1/df)
        noiseW = noiseW.to(s.device)
    if meanW is None:
        hypFunMean = compute_stimuli_hyp1f1(a=1, b=df/2+2, nc=nc)
        meanW = hypFunMean * (1/(df+2))
        meanW = meanW.to(s.device)
    # Compute the means of the second moments for each ctg of stimuli
    nClasses = torch.unique(ctgInd).size()[0]
    expectedSM = torch.zeros(nClasses, df, df, device=s.device)
    for cl in range(nClasses):
        # Get stim in this level
        mask = (ctgInd == cl)
        sLevel = s[mask,:]  # Stimuli of the same category
        nStimLevel = mask.sum()
        # Select the weights of the formula for this category
        noiseWMean = noiseW[mask].mean()  # Scalar by which to multiply identity term
        meanWStim = meanW[mask]  # Vector with scaling fators for each stimulus
        # Scale each stimulus by the sqrt of the outer prods weights
        stimScales = torch.sqrt(meanWStim/(nStimLevel))/sigma
        if normFactor is not None:
            stimScales = stimScales * torch.sqrt(normFactor[mask])
        scaledStim = torch.einsum('nd,n->nd', sLevel, stimScales)
        expectedSM[cl,:,:] = torch.einsum('nd,nb->db', scaledStim, scaledStim) + \
                torch.eye(df, device=s.device) * noiseWMean
    return expectedSM


##################################
##################################
#
## CALCULATE FILTER RESPONSE STATISTICS
#
##################################
##################################


def isotropic_ctg_resp_mean(s, sigma, f, normalization='broadband',
        ctgInd=None, sAmp=None, invNormMean=None):
    """Compute analytically the mean of filter responses to the noisy
        normalized stimuli of each category.
    ----------------
    Arguments:
    ----------------
      - s: Stimuli dataset (nStim x nDim)
      - sigma: Standard deviation of the isotropic noise (it is ignored if
          invNormMean is not None)
      - f: Filters (nFilt x nDim)
      - normalization: String indicating whether the filter response
          normalization is to be 'broadband' or 'narrowband'
      - ctgInd: Vector with category indices of stimuli s (length nStim)
      - sAmp: Optional to save compute when normalization='narrowband'.
          Stimulus amplitude spectrum. Should be computed with
          'au.compute_amplitude_spectrum(s)' (nStim x nDim)
      - invNormMean: Optional to save compute (broadband and narrowband).
          The expected values of the inverse of the norm for each stimulus s.
          Should be computed with the inv_ncx functions. (nStim)
    ----------------
    Outputs:
    ----------------
      - respMean: Response mean vector for each category. (nClasses x nFilt)
    """
    # 1) Housekeeping
    nStim = s.shape[0]
    # Adjust dimensions of filters to be row vectors
    if f.dim() == 1:
        f = f.unsqueeze(0)
    nFilt = f.shape[0]
    if ctgInd is None:
        ctgInd = torch.zeros(nStim)
    nClasses = len(ctgInd.unique())
    if normalization not in ['broadband', 'narrowband']:
        print('Error, normalization is neither narrowband nor broadband')
        return
    # Computation depends on normalization type
    if normalization == 'broadband':
        # 1) If broadband, a) calculate normalized stimulus means,
        # b) apply filters to the means
        #
        # a) Compute stimuli means
        if invNormMean is None:
            invNormMean = inv_ncx_batch(mu=s, sigma=sigma)
        stimMean = isotropic_mean_batch(s=s, sigma=sigma,
                invNormMean=invNormMean, ctgInd=ctgInd)
        # b) Apply filters to the means
        respMean = torch.einsum('cd,kd->ck', stimMean, f)
    elif normalization == 'narrowband':
        # 2) If narrowband, a) Calculate E(1/||X||),
        # b) Get filter responses to each normalized stimulus,
        # c) Compute similarity scores, d) Weight responses by similarity
        # scores, e) Compute the means of the responses
        # a) Get E(1/||X||)
        if invNormMean is None:
            invNormMean = inv_ncx_batch(mu=s, sigma=sigma)
        # b) Compute responses
        # Compute filter responses to normalized stimuli
        resp = torch.einsum('nd,kd,n->nk', s, f, invNormMean)
        # c) Compute similarities and normalization factor
        if sAmp is not None:
            similarities = compute_amplitude_similarity(s=sAmp,
                    f=f, stimSpace='fourier', filterSpace='signal')
        else:
            similarities = compute_amplitude_similarity(
                    s=torch.multiply(s, invNormMean.unsqueeze(1)),
                    f=f, stimSpace='signal', filterSpace='signal')
        normFactors = 1/similarities
        # d) Scale filter responses by stimulus similarity
        respScaled = resp * normFactors  # element-wise multiplication
        respMean = torch.zeros(nClasses, nFilt)
        for cl in range(nClasses):
            levelInd = [i for i, j in enumerate(ctgInd) if j == cl]
            respMean[cl,:] = torch.mean(respScaled[levelInd,:], dim=0)
    return respMean



def isotropic_ctg_resp_secondM(s, f, sigma, noiseW, meanW,
        normalization='broadband', ctgInd=None, normFactors=None):
    """Compute the second moment matrix of filter responses to
    the noisy (isotropic noise) normalized stimuli of each category, using
    an analytic formula.
    ----------------
    Arguments:
    ----------------
      - s: Stimuli dataset (nStim x nDim)
      - f: Filters (nFilt x nDim)
      - sigma: Standard deviation of the isotropic stimulus noise
      - noiseW: Vector with weights of the noise term in the analytic formulas
          (length nStim)
      - meanW: Vector with weights of the mean term in the analytic formulas
          (length nStim)
      - normalization: What normalization regime to use for the filter responses.
          String that is either 'broadband' or 'narrowband'.
      - ctgInd: Vector with category indices of stimuli s (length nStim).
          If not given, all stimuli are assigned to the same class.
      - normFactors: Factor by which to multiply the respone of each filter
          to each stimulus to implement narrowband normalization. It is
          given by 1/similarity for each filter/stimulus. (nStim x nFilt)
    ----------------
    Outputs:
    ----------------
      - expectedSM: Tensor with the second moment matrix of the filter
          responses for each category. (nClasses x nFilt x nFilt)
    """
    # Fill up vectors ctgInd with irrelevant values if not given
    nStim = s.shape[0]
    if ctgInd is None:
        ctgInd = torch.zeros(nStim)
    nFilt = f.shape[0]
    # Compute responses and filter inner products, which are
    # used in all cases
    responses = torch.einsum('kd,nd->nk', f, s)
    filterInnerProds = torch.einsum('kd,md->km', f, f)
    # If narrowband, compute similarities and weight responses
    if normalization=='narrowband':
        # Get the normalization factors given by similarity
        if normFactors is None:
            #### Add correction by inverse mean of s
            # Approximate normalization factor of s/||s+\sigma||
            invNormMean = inv_ncx_batch(mu=s, sigma=sigma)
            # Compute similarity scores
            similarities = compute_amplitude_similarity(
                    s=torch.multiply(s, invNormMean.unsqueeze(1)),
                    f=f, stimSpace='signal', filterSpace='signal')
            normFactors = 1/similarities
            # Adjust each response by the normalization factor
            responses = torch.einsum('nk,nk->nk', responses, normFactors)
    # Proceed the same for narrowband and broadband, after response weighting
    nClasses = np.unique(ctgInd).size
    expectedSM = torch.zeros(nClasses, nFilt, nFilt)
    for cl in range(nClasses):
        # Get index of stim in this level
        levelInd = [i for i, j in enumerate(ctgInd) if j == cl]
        nStimLevel = len(levelInd)
        # Get the weights for this class for stimuli
        meanWStim = meanW[levelInd]  # Vector with scaling fators for each stimulus
        # Get the noise term
        # First, compute the matrix of correlation of normalization factors
        if normalization=='narrowband':
            normFactorsCl = normFactors[levelInd,:] * \
                torch.sqrt(noiseW[levelInd]).unsqueeze(1)
            # Correlation of normalization factors
            normCorr = torch.einsum('nk,nm->km', normFactorsCl, normFactorsCl)
        else:
            normCorr = torch.ones((nFilt, nFilt))
        noiseTerm = normCorr * filterInnerProds * 1/nStimLevel
        # Scale each response by the normalizing factor
        scalingFactors = torch.sqrt(meanWStim/(nStimLevel))/sigma
        respScaled = torch.einsum('nd,n->nd', responses[levelInd,:], scalingFactors)
        expectedSM[cl,:,:] = torch.einsum('nd,nb->db', respScaled, respScaled) + \
                noiseTerm
    return expectedSM


##################################
##################################
#
## UTILITY FUNCTIONS
#
##################################
##################################


def compute_amplitude_similarity(s, f, stimSpace='signal',
        filterSpace='signal'):
    """Compute the similarity between the amplitude spectrum of
    stimuli in s, and of filters in f (i.e. s*f/(||s||*||f||).
    Return a matrix with similarity scores.
    ----------------
    Arguments:
    ----------------
      - s: Either the stimuli dataset, or their amplitude spectra. (nStim x nDim)
      - f: Either the filters, or their amplitude spectra. (nFilt x nDim)
      - stimSpace: String indicating if s is in signal space, or amplitude spectrum
      - filterSpace: String indicating if f is in signal space, or amplitude spectrum
    ----------------
    Outputs:
    ----------------
      - similarity: Similarity score matrix. (nStim x nFilt)
    """
    # Adjust dimensions of filters to be row vectors
    if f.dim() == 1:
        f = f.unsqueeze(0)
    # Get the Amplitude spectrum of the signal
    if stimSpace == 'signal':
        # Get amplitude spectrum
        sAmp = au.compute_amplitude_spectrum(s)
    else:
        sAmp = s
    # Get the Amplitude spectrum of the filter
    if filterSpace == 'signal':
        # Get amplitude spectrum
        fAmp = au.compute_amplitude_spectrum(f)
    else:
        fAmp = f
    # Get the normalization factor, i.e. inverse product of norms
    normFactor = torch.einsum('n,k->nk', 1/sAmp.norm(dim=1),
            1/fAmp.norm(dim=1))
    # Compute the Amplitude spectrum similarity
    similarity = torch.einsum('nd,kd,nk->nk', sAmp.float(), fAmp.float(),
        normFactor)
    return similarity


def secondM_2_cov(secondM, mean, nStim=None):
    """Convert matrices of second moments to covariances, by
    subtracting the product of the mean with itself.
    ----------------
    Arguments:
    ----------------
      - secondM: Second moment matrix. E.g. computed with
           'isotropic_ctg_resp_secondM'. (nClasses x nFilt x nFilt,
           or nFilt x nFilt)
      - mean: mean matrix. E.g. computed with 'isotropic_ctg_resp_mean'.
           (nClasses x nFilt, or nFilt)
    ----------------
    Outputs:
    ----------------
      - covariance: Covariance matrices. (nClasses x nFilt x nFilt)
    """
    if secondM.dim() == 2:
        secondM = secondM.unsqueeze(0)
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)
    # Get the multiplying factor to make covariance unbiased
    if nStim is None:
        unbiasingFactor = 1
    else:
        unbiasingFactor = nStim/(nStim-1)
    covariance = (secondM - torch.einsum('cd,cb->cdb', mean, mean)) * unbiasingFactor
    return covariance


