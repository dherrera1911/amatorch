# <markdowncell>

# # Approximating the covariance of noisy, normalized, neural responses
#
# Neuronal responses are noisy (i.e. there is variability in the
# responses to a repeated stimulus). Also, neuronal responses
# are usually divisively normalized by the pooled responses
# of other neurons. Thus, when modeling neuronal responses, it
# is convenient to apply these two pre-processing steps: additive
# noise and normalization.
#
# In the simplest case, noise is modeled as an isotropic Gaussian
# around the mean response, and normalization is broadband
# (i.e. the population response vector is divided by its norm).
# This is expressed in the following formula:
#
# $$c = \frac{s + \gamma}{||s + \gamma||}$$
#
# where $s \in \mathbb{R}^d$ is the expected value of the
# population response vector, with $d$ dimensions,
# $\gamma \sim \mathcal{N}(0,\mathbf{I}\sigma^2), \gamma \in \mathbb{R}^d$
# is a sample of multivariate white noise, and $c \in \mathbb{R}^d$ is the
# noisy, normalized, population response.
#
# When studying the statistics of responses of model neurons
# to natural signals (e.g. for building probabilistic response
# models), it is thus desirable to compute the statistics
# of the noisy, normalized responses.
#
# Here, we present analytic formulas to compute the covariances
# of noisy, normalized responses, from the noiseless 
# unnormalized responses. We first present the formula
# to obtain the second moment of the noisy normalized responses
# for single stimuli. Then, we use these to compute
# the covariance of the noisy normalized responses to a
# set of natural images

# <markdowncell>
#
# # Isotropic Gaussian noise and broadband normalization
#
# In the case of $c = \frac{s + \gamma}{||s + \gamma||}$
# with $\gamma \sim \mathcal{N}(0, \mathbf{I}\sigma^2)$, there
# is an exact formula to compute the second moment
# $\mathbb{E}_{\gamma}(cc^T)$
#
# \begin{equation}
#   \mathbb{E}_{\gamma}\left[cc^T \right] =
#   \frac{1}{d} {}_{1}F_1\left(1; \frac{d}{2}+1; \frac{-||\mu||^2}{2}\right)\mathbf{I} +
#   {}_{1}F_1\left(1; \frac{d}{2}+2; \frac{-||\mu||^2}{2}\right)\frac{1}{d+2}
#   \mu\mu^T
# \end{equation}
#
# where $\mu = \frac{1}{\sigma}s$ and ${}_{1}F_1\left(a; b; c\right)$
# is the hypergeometric confluent function
# (which is included in standard scientific computing packages). It can be seen
# that the result is a weigthed sum of the identity $\mathbf{I}$ and the
# outer product of the standardized mean, $\mu\mu^T$.
# The derivation is provided in the companion notes.
#
# We next show an implementation of this formula, and compare the
# analytic results to empirical results


# <codecell>
##############
#### IMPORT PACKAGES
##############
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mpm  # Important to use mpmath for hyp1f1, scipy blows up
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


# <codecell>
##############
#### DEFINE THE FUNCTION TO COMPUTE THE SECOND MOMENT OF
#### ISOTROPIC NOISE, BROADBAND NORM
##############
def isotropic_broadb_sm(s, sigma):
    """ Estimate the second moment of a noisy normalized stimulus,
    with isotropic white noise and broadband normalization.
    s: Stimulus mean. shape nDim
    sigma: Standar deviation of the isotropic noise
    """
    df = int(s.shape[0])  # Get number of dimensions
    sNorm = s/sigma  # Standardize the stimulus dividing it by sigma (\mu)
    nc = float(torch.sum(sNorm**2)) # non-centrality parameter, ||\mu||^2
    # Hypergeometric function for the term with the identity
    hypFunNoise = torch.tensor(float(mpm.hyp1f1(1, df/2+1, -nc/2)))
    # Hypergeometric function for the term with the mean
    hypFunMean = torch.tensor(float(mpm.hyp1f1(1, df/2+2, -nc/2))) 
    # Get the outer product of the normalized stimulus, and multiply by weight
    meanTerm = torch.einsum('a,b->ab', sNorm, sNorm) * 1/(df+2) * hypFunMean
    # Multiply the identity matrix by weighting term 
    noiseTerm = torch.eye(df) * 1/df * hypFunNoise
    # Add the two terms
    expectedCov = (meanTerm + noiseTerm)
    return expectedCov

# <codecell>
##############
#### COMPUTE ANALYTIC AND EMPIRICAL SECOND MOMENTS
##############
# All combinations of dimensions and noise below will be used
nDim = torch.tensor([20, 50, 200]) # Vector with number of dimensions to use
sigmaVec = torch.tensor([0.1, 0.5, 1, 2, 5]) # Vector with sigma values to use
nFits = len(nDim) * len(sigmaVec)

nSamplesRef = torch.tensor(2*10**5) # N Samples for reference empirical distribution
nSamplesLow = torch.tensor(1000) # N Samples for noisy empirical estimation

smDict = {'nDim': torch.zeros(nFits), 'sigma': torch.zeros(nFits),
        'smAnalytic': [], 'smEmpRef': [], 'smEmpLow': []}
ii=0
for d in range(len(nDim)):
    for n in range(len(sigmaVec)):
        df = nDim[d]  # Dimensions of vector
        sigma = sigmaVec[n]   # Standard deviation of noise
        # Use a sine function as the mean of the distribution
        s = torch.sin(torch.linspace(0, 2*torch.pi, df))
        # Store noise and dimension values
        smDict['nDim'][ii] = df
        smDict['sigma'][ii] = sigma
        ### Calculate analytic second moment:
        smAnalytic = isotropic_broadb_sm(s, sigma)
        ### Calculate empirical reference distribution:
        # Initialize distribution of samples
        Cov = torch.eye(df)*(sigma**2)
        # distribution of x = s + \gamma
        xDist = MultivariateNormal(loc=s, covariance_matrix=Cov)
        # Make random samples
        xSamples = xDist.rsample([nSamplesRef]) # for reference value
        xSamplesLow = xDist.rsample([nSamplesLow]) # for noisy estimation
        # Get normalizing factors for the samples
        normRef = (torch.norm(xSamples, dim=1)**2)
        normLow = (torch.norm(xSamplesLow, dim=1)**2)
        # Compute empirical second moments
        smEmpRef = torch.einsum('ni,n,nj->ij', xSamples, 1/normRef, xSamples) / nSamplesRef
        smEmpLow = torch.einsum('ni,n,nj->ij', xSamplesLow, 1/normLow, xSamplesLow) / nSamplesLow
        # Store second moments in dictionary
        smDict['smAnalytic'].append(smAnalytic)
        smDict['smEmpRef'].append(smEmpRef)
        smDict['smEmpLow'].append(smEmpLow)
        ii = ii+1


# <markdowncell>
#
# Below we plot the individual elements of the covariance
# matrices, obtained empirically through a large number of
# samples (X axis), obtained empirically with a small number
# of samples (red dots), or obtained analytically with the
# formula above (blue dots).
#
# We will see that the empirical estimates of the covariance
# can be very noisy, but that the analytic expression
# is practically identical to the large-sample reference empirical
# distribution.

# <codecell>
##############
#### COMPARE ANALYTIC METHOD TO LOW-SAMPLE EMPIRICAL ESTIMATION
##############

### Plot analytical vs empirical covariances
nCols = len(nDim)
nRows = len(sigmaVec)

ii=0
for c in range(nCols):
    for r in range(nRows):
        # Extract values for this plot
        sigma = smDict['sigma'][ii]
        df = smDict['nDim'][ii]
        smEmpRef = smDict['smEmpRef'][ii].reshape(int(df**2))
        smEmpLow = smDict['smEmpLow'][ii].reshape(int(df**2))
        smAnalytic = smDict['smAnalytic'][ii].reshape(int(df**2))
        ax = plt.subplot(nCols, nRows, ii+1)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        # scatter the values
        plt.scatter(smEmpRef, smEmpLow, c='red', label=f'{nSamplesLow} samples', s=1.5)
        plt.scatter(smEmpRef, smAnalytic, color='blue', label='Analytic', s=1.5)
        # Add identity line
        ax.axline((0,0), slope=1, color='black')
        # Add names only on edge panels
        if c==(nCols-1):
            plt.xlabel('Empirical reference')
        if r==0:
            plt.ylabel('Estimation')
        plt.title(f'd={int(df)}, $\sigma$ = {sigma:.1f}', fontsize=11)
        if c==(nCols-1) and r==(nRows-1):
            handles, labels = ax.get_legend_handles_labels()
        ii = ii+1
# Add color code legend
fig = plt.gcf()
(lines, labels) = ax.get_legend_handles_labels()
lines = lines[0:2]
labels = labels[0:2]
fig.legend(lines, labels, loc='upper right')
fig.set_size_inches(10,6)
plt.show()


# <codecell>
### For better visualization of the results, we
### view the 3 second moment matrices for one of the cases

# Select the noise and dimensions level
noiseInd = 3
dimInd = 2

# Look for the index in the dictionary matching those levels
ind = torch.logical_and(smDict['nDim']==nDim[dimInd], \
        smDict['sigma']==sigmaVec[noiseInd])
ind = np.flatnonzero(ind)[0]

# Plot the matrices
plt.title('Second moment matrices')
plt.subplot(1,3,1)
plt.imshow(smDict['smAnalytic'][ind])
plt.title(f'Analytic', fontsize=11)
plt.subplot(1,3,2)
plt.title(f'Empirical ref', fontsize=11)
plt.imshow(smDict['smEmpRef'][ind])
plt.subplot(1,3,3)
plt.title(f'Empirical, {int(nSamplesLow)} samples', fontsize=11)
plt.imshow(smDict['smEmpLow'][ind])
fig = plt.gcf()
fig.suptitle(f'Second moments d={nDim[dimInd]}, $\sigma$={sigmaVec[noiseInd]}')
plt.setp(fig.get_axes(), xticks=[], yticks=[])
fig.set_size_inches(7,3)
plt.show()


# <markdowncell>
#
# # Non-isotropic Gaussian noise, flexible normalization
#
# A more general model for noisy normalized stimuli involves
# non-isotropic Gaussian noise, $\gamma \sim \mathcal{N}(0, \Psi)$
# where $\Psi$ is any symmetric positive-definite matrix, and
# flexible normalization, $\frac{1}{g_{s,f}||s + \gamma||}$,
# where $g_{s,f}$ is a normalizing factor that depends on the mean
# of the stimulus and a linear filter.
# 
# To estimate the second moment of a noisy, normalized stimulus
# under these conditions, the following approximation can be used
# (derived in the companion notes)
# 
# $$\mathbb{E}_{\gamma}\left( \frac{XX^T}{g_{s,f}^2||X||^2} \right) \approx
#   \frac{\mu_N}{\mu_D} \odot \left( 1 - \frac{\Sigma^{N,D}}{\mu_N\mu_D} + \frac{Var(D)}{\mu_D^2} \right)$$
# 
# where $X \sim \mathcal{N}(s, \Psi)$, $\odot$ indicates matrix-wise
# multiplication, divisions between matrices are taken element-wise, and
# $$\mu_N = \Psi + s s^T $$
# $$\mu_{D} = g_{s,f}^2 (tr(\Psi) + ||s||^2)$$
# $$Var(D) = g_{s,f}^4 \left(2 tr(\Psi^2) + 4 s^T \Psi s\right)$$
# $$\Sigma^{N,D} = 2 g_{s,f}^2 \left[\Psi \Psi + s s^T \Psi +  \Psi s s^T \right]$$
#
# Note that while $\mu_N \in \mathbb{R}^{d\times d}$ and
# $\Sigma^{N,D} \in \mathbb{R}^{d\times d}$,
# $\mu_D \in \mathbb{R}$ and $Var(D) \in \mathbb{R}$.
#
# In the notation above, $N$ indicates moments of the numerator
# inside the expectation (i.e. products of the form $X_iX_j$)
# and $D$ refers to moments of the denominator (i.e.
# $g_{s,f}||X||^2$).
#
# Below, we test the performance of this approximation to recover
# response second moment under these noise and normalization conditions


# <codecell>
##############
#### DEFINE THE FUNCTION TO COMPUTE THE SECOND MOMENT OF
#### ISOTROPIC NOISE, BROADBAND NORM
##############
def general_flexible_sm(s, sigmaCov, normScale):
    """ Estimate the second moment of a noisy normalized stimulus,
    with any Gaussian noise covariance and flexible normalization.
    s: Stimulus mean. shape nDim
    sigmaCov: Covariance matrix of the noise. shape nDim x nDim
    normScale: Scalar that scales normalization.
    """
    # Compute some repeated quantities
    ssOuter = torch.einsum('i,j->ij', s, s) # outer prod of stim with itself
    sigmaCov2 = torch.matmul(sigmaCov,sigmaCov) # square of covariance matrix
    # Mean of numerator
    mun = sigmaCov + ssOuter
    # Mean of denominator
    mud = normScale**2 * (sigmaCov.trace() + torch.einsum('i,i->', s, s))
    # Variance of denominator
    vard = normScale**4*(2*torch.trace(sigmaCov2) + \
            4*torch.einsum('i,ij,j->', s, sigmaCov, s))
    # Correlation between numerator and denominator 
    covnd = 2 * normScale**2 * (sigmaCov2 + torch.matmul(ssOuter, sigmaCov) + \
        torch.matmul(sigmaCov, ssOuter))
    # Expected value of ratio quadratic forms
    secondMoment = mun/mud * (1 - covnd/(mun*mud) + vard/(mud**2))
    return secondMoment


# <codecell>
##############
#### COMPUTE ANALYTIC AND EMPIRICAL SECOND MOMENTS
##############

# All combinations of dimensions and noise below will be used
nDim = torch.tensor([10, 50, 200]) # Vector with number of dimensions to use
sigmaVec = torch.tensor([0.05, 0.2, 1, 2, 4]) # Vector with sigma values to use to scale cov
gVec = torch.tensor([0.7]) # normalizing factor

covDiag = False

nFits = len(nDim) * len(sigmaVec) * len(gVec)

nSamplesRef = torch.tensor(2*10**5) # N Samples for reference empirical distribution
nSamplesLow = torch.tensor(1000) # N Samples for noisy empirical estimation

smDict = {'nDim': torch.zeros(nFits), 'sigma': torch.zeros(nFits), 'g': torch.zeros(nFits),
        'sigmaCov': [], 'smAnalytic': [], 'smEmpRef': [], 'smEmpLow': []}
ii=0
for d in range(len(nDim)):
    for n in range(len(sigmaVec)):
        for j in range(len(gVec)):
            df = int(nDim[d])  # Dimensions of vector
            sigma = sigmaVec[n]   # Standard deviation of noise
            g = gVec[j]
            # Use a sine function as the mean of the distribution
            s = torch.sin(torch.linspace(0, 2*torch.pi, df))
            # Store noise and dimension values
            smDict['nDim'][ii] = df
            smDict['sigma'][ii] = sigma
            smDict['g'][ii] = g
            # Make a random covariance matrix and scale by sigma
            # Make random matrix + cosine to make symmetric matrix
            if not covDiag:
                #c1 = torch.randn(int(df), int(df)) + \
                #        torch.cos(torch.linspace(0, 3*torch.pi, df)).unsqueeze(1)
                #offDiag = torch.matmul(c1, c1.transpose(0,1))
                #diagW = torch.rand(1)
                #sigmaCov = (offDiag * (1-diagW) + torch.eye(df)*diagW) * sigma
                c1 = torch.randn(int(df), int(df))
                offDiag = torch.cov(c1)
                diagW = torch.rand(1)
                sigmaCov = (offDiag * (1-diagW) + torch.eye(df)*diagW) * sigma
            else:
                sigmaCov = torch.diag((torch.rand(df)+0.5)*2-1)
            smDict['sigmaCov'].append(sigmaCov)
            ### Calculate analytic second moment:
            smAnalytic = general_flexible_sm(s, sigmaCov, g)
            ### Calculate empirical reference distribution:
            # Initialize distribution of samples
            # distribution of x = s + \gamma
            xDist = MultivariateNormal(loc=s, covariance_matrix=sigmaCov)
            # Make random samples
            xSamples = xDist.rsample([int(nSamplesRef)]) # for reference value
            xSamplesLow = xDist.rsample([int(nSamplesLow)]) # for noisy estimation
            # Get normalizing factors for the samples
            normRef = (torch.norm(xSamples, dim=1)*g)**2
            normLow = (torch.norm(xSamplesLow, dim=1)*g)**2
            # Compute empirical second moments
            smEmpRef = torch.einsum('ni,n,nj->ij', xSamples, 1/normRef, xSamples) / nSamplesRef
            smEmpLow = torch.einsum('ni,n,nj->ij', xSamplesLow, 1/normLow, xSamplesLow) / nSamplesLow
            # Store second moments in dictionary
            smDict['smAnalytic'].append(smAnalytic)
            smDict['smEmpRef'].append(smEmpRef)
            smDict['smEmpLow'].append(smEmpLow)
            ii = ii+1


# <markdowncell>
#
# Below we compare empirical and analytic results to an
# empirical reference, like in previous section.
#
# We see that the empirical estimates of the covariance
# can be noisy, and that the analytic expression works
# well for most cases

# <codecell>
##############
#### COMPARE ANALYTIC METHOD TO LOW-SAMPLE EMPIRICAL ESTIMATION
##############

# <codecell>
### Plot analytical vs empirical covariances
nRows = len(nDim)
nCols = len(sigmaVec)

ii=0
for r in range(nRows):
    for c in range(nCols):
        # Extract values for this plot
        sigma = smDict['sigma'][ii]
        df = smDict['nDim'][ii]
        smEmpRef = smDict['smEmpRef'][ii].reshape(int(df**2))
        smEmpLow = smDict['smEmpLow'][ii].reshape(int(df**2))
        smAnalytic = smDict['smAnalytic'][ii].reshape(int(df**2))
        ax = plt.subplot(nRows, nCols, ii+1)
        # scatter the values
        plt.scatter(smEmpRef, smEmpLow, c='red', label=f'{nSamplesLow} samples', s=2)
        plt.scatter(smEmpRef, smAnalytic, color='blue', label='Analytic', s=2)
        # Add identity line
        ax.axline((0,0), slope=1, color='black')
        # Add names only on edge panels
        if r==(nRows-1):
            plt.xlabel('Empirical reference')
        else:
            ax.set_xticks([], [])
        if c==0:
            plt.ylabel('Estimation')
        else:
            ax.set_yticks([], [])
        plt.title(f'd={int(df)}, $\sigma$ = {sigma:.2f}', fontsize=11)
        if r==(nRows-1) and c==(nCols-1):
            handles, labels = ax.get_legend_handles_labels()
        ii = ii+1
# Add color code legend
fig = plt.gcf()
(lines, labels) = ax.get_legend_handles_labels()
lines = lines[0:2]
labels = labels[0:2]
fig.legend(lines, labels, loc='upper right')
fig.set_size_inches(10,6)
plt.show()


# <codecell>
### For better visualization of the results, we
### view the 3 second moment matrices for one of the cases
# Select the noise and dimensions level
noiseInd = 2
dimInd = 2

# Look for the index in the dictionary matching those levels
ind = torch.logical_and(smDict['nDim']==nDim[dimInd], \
        smDict['sigma']==sigmaVec[noiseInd])
ind = np.flatnonzero(ind)[0]

# Plot the matrices
plt.subplot(1,3,1)
plt.imshow(smDict['smAnalytic'][ind])
plt.title(f'Analytic', fontsize=11)
plt.subplot(1,3,2)
plt.title(f'Empirical ref', fontsize=11)
plt.imshow(smDict['smEmpRef'][ind])
plt.subplot(1,3,3)
plt.title(f'Empirical {int(nSamplesLow)} samples', fontsize=11)
plt.imshow(smDict['smEmpLow'][ind])
fig = plt.gcf()
fig.suptitle(f'Second moments d={int(nDim[dimInd])}, $\sigma$={sigmaVec[noiseInd]:.2f}')
plt.setp(fig.get_axes(), xticks=[], yticks=[])
fig.set_size_inches(7,3)
plt.show()


# <codecell>
### Let's see the random covariance matrices that we sampled for each case
ii = 0
for r in range(nRows):
    for c in range(nCols):
        # Extract values for this plot
        sigma = smDict['sigma'][ii]
        df = smDict['nDim'][ii]
        sigmaCov = smDict['sigmaCov'][ii]
        ax = plt.subplot(nRows, nCols, ii+1)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        plt.imshow(sigmaCov)
        plt.title(f'd={int(df)}, $\sigma$ = {sigma:.2f}', fontsize=11)
        ii = ii+1
fig = plt.gcf()
fig.suptitle(f'Noise covariances for each dimension and $\sigma$')
plt.setp(fig.get_axes(), xticks=[], yticks=[])
fig.set_size_inches(10,9)
plt.show()


#
#########################
##### Do empirical vs analytic for the 3D speed dataset
#########################
#import scipy.io as spio
#import numpy as np
#import torch
#import matplotlib.pyplot as plt
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.utils.data import TensorDataset, DataLoader
#
## Load ama struct from .mat file into Python
#data = spio.loadmat('./data/amaInput_12dir_3speed_0stdDsp_train.mat')
## Extract contrast normalized, noisy stimulus
#s = data.get("Iret")
#s = torch.from_numpy(s)
#s = s.transpose(0,1)
#s = s.float()
## Extract the vector indicating category of each stimulus row
#ctgInd = data.get("ctgIndMotion")
#ctgInd = torch.tensor(ctgInd)
#ctgInd = ctgInd.flatten()
#ctgInd = ctgInd-1       # convert to python indexing (subtract 1)
#ctgInd = ctgInd.type(torch.LongTensor)  # convert to torch integer
## Extract the values of the latent variable
#ctgVal = data.get("Xmotion")
#ctgVal = torch.from_numpy(ctgVal)
#nPixels = int(s.shape[1]/2)
## Noise parameters
#filterSigma = 0.23
#pixelSigma = 0.00075
#
#def contrast_stim(s):
#    nPixels = s.shape[1]
#    sMean = torch.mean(s, axis=1)  # Mean intensity of each stimulus, not mean of dataset
#    sContrast = torch.einsum('nd,n->nd', (s - sMean.unsqueeze(1)), 1/sMean)
#    return sContrast
#
#s = contrast_stim(s)
#
#
#def analytic_noisy_cov(s, noise=0):
#    for 
#    invNormAn = inv_ncx2(df=df, nc=nc) * (1/sigma**2)
#
#def analytic_noisy_cov(s, ctgInd):
#    nDim = s.shape[1]
#    nClasses = ctgInd.unique().shape[0]
#    # Compute the conditional statistics of the stimuli
#    stimCovs = torch.zeros(nClasses, nDim, nDim)
#    stimMeans = torch.zeros(nClasses, nDim)
#    for cl in range(nClasses):
#        levelInd = [i for i, j in enumerate(ctgInd) if j == cl]
#        sLevel = sAll[levelInd, :]
#        self.stimCovs[cl, :, :] = torch.cov(sLevel.transpose(0,1))
#        self.stimMeans[cl, :] = torch.mean(sLevel, 0)
#
#
