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
    nc = float(torch.sum(sNorm**2))  # non-centrality parameter, ||\mu||^2
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

# Number of samples
nSamplesRef = torch.tensor(2*10**5) # N Samples for reference empirical distribution
nSamplesLow = torch.tensor(1000) # N Samples for noisy empirical estimation

smDict = {'nDim': torch.zeros(nFits), 'sigma': torch.zeros(nFits),
    'smAnalytic': [], 'smEmpRef': [], 'smEmpLow': []}
ii = 0
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

ii = 0
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
# where $\Psi$ is any symmetric positive-semi-definite matrix, and
# flexible normalization, $\frac{1}{g_{s,f}||s + \gamma||}$,
# where $g_{s,f}$ is a normalizing factor that depends on the
# unnoisy, unnormalized stimulus $\mathbf{s}$, and on a linear
# filter $\mathbf{f}$.
#
# To estimate the second moment of a noisy, normalized stimulus
# under these conditions, the following approximation can be used
# (derived in the companion notes)
#
# \begin{equation}
# \mathbb{E}_{\gamma}\left( \frac{XX^T}{||X||^2} \right) \approx
#     \frac{1}{g_{s,f}^2} \frac{\mathbf{\mu}_N}{\mu_D} \odot \left( 1 -
#     \frac{\mathbf{\Sigma}^{N,D}}{\mathbf{\mu}_N\mu_D} + \frac{Var(D)}{\mu_D^2} \right)
# \end{equation}
#
# where $X \sim \mathcal{N}(\mathbf{s}, \Psi)$, $\odot$ indicates matrix-wise
# multiplication, divisions between matrices are taken element-wise, and
# $$\mu_N = \Psi + s s^T $$
# $$\mu_{D} = (tr(\Psi) + ||s||^2)$$
# $$Var(D) = \left(2 tr(\Psi^2) + 4 s^T \Psi s\right)$$
# $$\Sigma^{N,D} = 2 \left[\Psi \Psi + s s^T \Psi +  \Psi s s^T \right]$$
#
# Note that while $\mu_N \in \mathbb{R}^{d\times d}$ and
# $\Sigma^{N,D} \in \mathbb{R}^{d\times d}$,
# $\mu_D \in \mathbb{R}$ and $Var(D) \in \mathbb{R}$.
#
# In the notation above, $N$ indicates moments of the numerator
# inside the expectation (i.e. products of the form $X_iX_j$)
# and $D$ refers to moments of the denominator (i.e.
# $||X||^2$).
#
# We note that the stimulus-specific normalization factor
# $\frac{1}{g_{s,f}^2}$ does not depend on $\gamma$, and thus it can just be
# taken out of the expectation. Thus, stimulus-specific normalization
# can also be applied to the isotropic noise case from previous section.
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
    ssOuter = torch.einsum('i,j->ij', s, s) # outer prod of stim with from previous section
    sigmaCov2 = torch.matmul(sigmaCov,sigmaCov) # square of covariance matrix
    # Mean of numerator
    mun = sigmaCov + ssOuter
    # Mean of denominator
    mud = (sigmaCov.trace() + torch.einsum('i,i->', s, s))
    # Variance of denominator
    vard = (2*torch.trace(sigmaCov2) + 4*torch.einsum('i,ij,j->', s, sigmaCov, s))
    # Correlation between numerator and denominator 
    covnd = 2 * (sigmaCov2 + torch.matmul(ssOuter, sigmaCov) + \
        torch.matmul(sigmaCov, ssOuter))
    # Expected value of ratio quadratic forms
    secondMoment = (1/normScale**2) * mun/mud * \
        (1 - covnd/(mun*mud) + vard/(mud**2))
    return secondMoment


# <codecell>
##############
#### DEFINE A FUNCTION TO CREATE DIFFERENT TYPES OF COVARIANCE MATRICES
##############

# Make a function to create custom covariance matrices
def make_cov(covType, sigmaScale, df, decay=1, baseline=0):
    """ Generate a covariance matrix with specified properties.
    covType: Indicates the type of process that the covariance
        describes. Is a string that can take values:
        -'random': Random diagonal and off-diagonal elements.
        -'diagonal': Random diagonal elements. Off-diagonal elements are 0.
        -'scaled': Diagonal elements = sigmaScale + baseline. Off diagonal
            elements are 0. Can be used for Poisson-like noise.
        -'decaying': Diagonal elements = sigmaScale. Off-diagonal elements
            decay exponentially with distance to diagonal
    sigmaScale: For most covTypes, it is a scalar that multiplies the
        resulting covariance matrix. For covType='scaled', it is a vector
        that is used as the diagonal (+ baseline).
    df: Dimensionality of the matrix.
    decay: Rate of decay of the 'decaying' type of matrix. Is a scalar.
    baseline: Constant added to 'scaled' covariance matrix diagonal elements.
    """
    if covType=='random':
        isPD = False
        while not isPD:
            randMat = torch.randn(int(df), int(df))  # Make random matrix
            covRand = torch.cov(randMat)  # Turn it into positive-semidefinite matrix
            diagW = torch.rand(1)  # Sample a relative weight of diagonal-off diagonal elements
            sigmaCov = ((1-diagW) * covRand + diagW * torch.diag(covRand.diag())) * sigma
            eigVals = torch.real(torch.linalg.eigvals(sigmaCov))
            isPD = all(eigVals > 0)
    if covType=='diagonal':
        sigmaCov = torch.diag((torch.rand(df)+0.5)*2-1)
    if covType=='scaled':
        sigmaCov = torch.diag(torch.abs(sigmaScale)+baseline)
    if covType=='decaying':
        cov = torch.zeros((df, df))
        indRow = torch.reshape(torch.linspace(0, 1, df), (df,1)).repeat(1,df)
        indCol = torch.reshape(torch.linspace(0, 1, df), (1,df)).repeat(df,1)
        diagDist = torch.abs(indRow - indCol)
        sigmaCov = torch.exp(-(diagDist*decay)) * sigmaScale
    return sigmaCov


# <codecell>
##############
#### COMPUTE ANALYTIC AND EMPIRICAL SECOND MOMENTS
##############

# All combinations of dimensions and noise below will be used
nDim = torch.tensor([10, 50, 200])  # Vector with number of dimensions to use
sigmaVec = torch.tensor([0.2, 1, 3])  # Vector with sigma values to use to scale cov
gVec = torch.tensor([0.7]) # normalizing factor

# covariance parameters
covType = 'decaying'
decay = 5
baseline = 0.05

# Number of samples
nSamplesRef = torch.tensor(10**6)  # N Samples for reference empirical distribution
nSamplesLow = torch.tensor(1000)     # N Samples for noisy empirical estimation

# Number of different random variables to test
nFits = len(nDim) * len(sigmaVec) * len(gVec)

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
            # Make a random covariance matrix as requested
            if covType=='scaled':
                sigma = sigma*s
            sigmaCov = make_cov(covType=covType, sigmaScale=sigma, df=df,
                decay=decay, baseline=baseline)
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
        df = int(smDict['nDim'][ii])
        nonRepeatedInds = torch.nonzero(torch.tril(torch.ones(df,df)).reshape(df**2))
        smEmpRef = smDict['smEmpRef'][ii].reshape(df**2)[nonRepeatedInds]
        smEmpLow = smDict['smEmpLow'][ii].reshape(df**2)[nonRepeatedInds]
        smAnalytic = smDict['smAnalytic'][ii].reshape(df**2)[nonRepeatedInds]
        ax = plt.subplot(nRows, nCols, ii+1)
        # scatter the values
        plt.scatter(smEmpRef, smEmpLow, c='red', label=f'{nSamplesLow} samples',
                s=1.5, alpha=0.2)
        plt.scatter(smEmpRef, smAnalytic, color='blue', label='Analytic',
                s=1, alpha=0.2)
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
fig.set_size_inches(16,14)
plt.show()

# <markdowncell>
#
# Next, we show, for one of the examples given above (i.e.
# one noise level, and one dimensionality) the empirical
# reference matrix, our approximation, and the low-sample
# empirical approximation. We also show a matrix with the
# differences between analytic and empirical approximation
# and the reference.

# <codecell>
### For better visualization of the results, we
### view the 3 second moment matrices for one of the cases
# Select the noise and dimensions level
noiseInd = 1
dimInd = 1

# Look for the index in the dictionary matching those levels
ind = torch.logical_and(smDict['nDim']==nDim[dimInd], \
        smDict['sigma']==sigmaVec[noiseInd])
ind = np.flatnonzero(ind)[0]

maxErr1 = torch.max(smDict['smEmpRef'][ind]-smDict['smAnalytic'][ind])
maxErr2 = torch.max(smDict['smEmpRef'][ind]-smDict['smEmpLow'][ind])
maxErr = max((maxErr1, maxErr2))

# Plot the matrices
# Analytic second moment matrix
plt.subplot(2,3,1)
plt.imshow(smDict['smAnalytic'][ind])
plt.title(f'Analytic', fontsize=11)
# Diff Reference - Empirical
plt.subplot(2,3,4)
plt.imshow(smDict['smEmpRef'][ind]-smDict['smAnalytic'][ind], cmap='bwr')
plt.clim(-maxErr, maxErr)
plt.title(f'Reference - Analytic', fontsize=11)
#  Empirical reference
plt.subplot(2,3,2)
plt.imshow(smDict['smEmpRef'][ind])
plt.title(f'Empirical ref', fontsize=11)
#  Empirical low sample
plt.subplot(2,3,3)
plt.imshow(smDict['smEmpLow'][ind])
plt.title(f'Empirical {int(nSamplesLow)} samples', fontsize=11)
#  Diff Reference - :ow sample
plt.subplot(2,3,6)
plt.imshow(smDict['smEmpRef'][ind] - smDict['smEmpLow'][ind], cmap='bwr')
plt.clim(-maxErr, maxErr)
plt.title(f'Reference - Low sample', fontsize=11)
fig = plt.gcf()
fig.suptitle(f'Second moments d={int(nDim[dimInd])}, $\sigma$={sigmaVec[noiseInd]:.2f}')
plt.setp(fig.get_axes(), xticks=[], yticks=[])
fig.set_size_inches(9,8)
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
fig.set_size_inches(14,12)
plt.show()



# <markdowncell>
#
# # Covariance of naturalistic image dataset

# Next, we move beyond the single, artificial stimulus that has
# been used so far, and apply the analytic second moment estimation
# to estimate the covariance of a real image dataset.
#
# We use a naturalistic dataset of binocular video patches of 3D motion.
# The dataset is composed of a set of non-noisy, unnormalized contrast
# stimuli, $\mathbf{s}_v \in \mathbb{R}^d$, where
# $v \in \{1, ..., n\}$, and $n$ is the number of stimuli in the dataset.
# The stimuli in our dataset are 1D binocular videos. They contain the
# input corresponding to two retinas, with a number nPixels of horizontal
# pixels, and with a number nTimesteps of time steps.
#
# We want to estimate the covariance of the dataset of noisy, normalized
# stimuli (as defined above)
# $\mathbf{c}_v \in \mathbb{R}^d$ with $v \in \{1, ..., n\}$, and
# isotropic noise $\gamma \sim \mathcal{N}(0,\mathbf{I}\sigma^2)$.
# The expression for the covariance can be reduced to the
# mean of the covariances of the noisy stimuli across the dataset
# (see accompanying notes for the derivation)
#
# \begin{equation}
#     \mathbb{E}_{\gamma,\mathbf{s}}\left(\mathbf{c}\mathbf{c}^T\right) =
#     \frac{1}{n} \sum_{v=1}^{v=n} \mathbb{E}_{\gamma}\left(\mathbf{c}_v\mathbf{c}_v^T\right)
# \end{equation}
#
# In this section, we use the formula for the expected
# second moment under isotropic noise and broadband
# normalization (presented in section 1 of the notebook)
# to estimate the second moment of each stimulus,
#$\mathbb{E}_{\gamma}\left(\mathbf{c}_v\mathbf{c}_v^T\right)
# and then put these together into the full dataset second moment as
# shown above.
#
# We also compare the results of our analytic calculation to
# the result of low-sample empirical estimation of the second moment

# <codecell>
#########################
# IMPORT PACKAGES AND AMA UTILITIES
#########################
import scipy.io as spio
import time
import scipy as sp

# <codecell>
# COMMENT THIS CELL FOR GOOGLE COLAB EXECUTION
#import ama_library.ama_utilities as au

# <codecell>
#### UNCOMMENT THIS CELL FOR GOOGLE COLAB EXECUTION
!pip install geotorch
import geotorch
!pip install git+https://github.com/dherrera1911/accuracy_maximization_analysis.git
import ama_library.ama_utilities as au
!wget -O ./data/amaInput_12dir_3speed_0stdDsp_train.mat https://drive.google.com/file/d/1m7BXFZFe0ppsHhURFbhqaCqHR3vTbD6V/view?usp=sharing

# <codecell>
#########################
# IMPORT AND PREPROCESS THE STIMULUS DATASET
#########################
# Load ama struct from .mat file into Python
data = spio.loadmat('./data/amaInput_12dir_3speed_0stdDsp_train.mat')
# Extract contrast normalized, noisy stimulus
s = data.get("Iret")
s = torch.from_numpy(s)
s = s.transpose(0,1)
s = s.float()
# We turn the images into contrast stimuli
sWeb = au.contrast_stim(s)
# Extract the vector indicating category of each stimulus row
ctgInd = data.get("ctgIndMotion")
ctgInd = torch.tensor(ctgInd)
ctgInd = ctgInd.flatten()
ctgInd = ctgInd-1       # convert to python indexing (subtract 1)
ctgInd = ctgInd.type(torch.LongTensor)  # convert to torch integer
nCtg = int(ctgInd.max()+1)
# Extract the values of the latent variable
ctgVal = data.get("Xmotion")
ctgVal = torch.from_numpy(ctgVal)
# Extract some properties of the dataset
htz = data.get('smpPerSec')
nTimesteps = int(data.get('durationMs')/(1000)*htz)
nPixels = len(data.get('smpPosDegX'))
nStim = sWeb.shape[0]
df = s.shape[1]

# <codecell>
#########################
# VISUALIZE A STIMULUS
#########################
nRand = torch.randint(high=nStim-1, size=(1,1))
ax = au.view_filters_bino_video(sWeb[nRand,:].unsqueeze(0),
        frames=nTimesteps, pixels=nPixels)
ax.axes.xaxis.set_visible(True)
ax.axes.yaxis.set_visible(True)
plt.xlabel('Pixels')
plt.ylabel('Time')
plt.title('Left eye                   Right eye')
plt.show()


# <codecell>
#########################
# SELECT PARAMETERS FOR THE COVARIANCE ESTIMATION PROCEDURE
#########################
# Choose the std for the isotropic gaussian noise
pixelSigma = 0.2
# Choose one category of the random variable whose second moment
# to estimate
ctg = 14
# Number of samples per stimulus for the empirical 'true' reference
samplesPerStimRef = 700
# Number of samples per stimulus for the empirical low-samples reference
samplesPerStimLow = 1  # Samples per each stim, low sample estimate
#
# Extract the stimuli of this category
sCtg = sWeb[torch.nonzero(ctgInd==ctg),:]. squeeze(1)
# Get number of total stim and samples
nCtgStim = sCtg.shape[0]
nSamplesRef = nCtgStim * samplesPerStimRef
nSamplesLow = nCtgStim * samplesPerStimLow
# Generate the Covariance matrix
noiseCov = torch.eye(df)*pixelSigma**2

# <codecell>
#########################
# COMPUTE ANALYTIC ESTIMATE OF SECOND MOMENT FOR THE CATEGORY
#########################
#
# Get analytic estimate and time it
start = time.time()
analyticCov = au.isotropic_broadb_sm_batch(sCtg, sigma=pixelSigma)
end = time.time()
print(f'Analytic took: {end-start}')


# <codecell>
#########################
# COMPUTE EMPIRICAL ESTIMATES
#########################
#
# Initialize a torch multivariante normal distribution for
# the noise with the required covariance
gamma = MultivariateNormal(loc=torch.zeros(df),
        covariance_matrix=noiseCov)
#
# Make the samples of the noisy, normalized dataset, with
# the reference number of samples
start = time.time()
start = time.time()
gamma.rsample([int(nSamplesRef)])
end = time.time()
end - start
xSamples = sCtg.repeat(samplesPerStimRef,1) + gamma.rsample([int(nSamplesRef)])
# Get normalization factor for each sample
xSamplesNorm = xSamples.norm(dim=1)
# Normalize each random sample
xSamples = torch.einsum('nd,n->nd', xSamples, 1/xSamplesNorm)
# Compute the second moment matrix of the samples
refCov = torch.einsum('nd,nb->db', xSamples, xSamples) * 1/nSamplesRef
end = time.time()
print(f'Reference took: {end-start}')
#
# Do the same for the low-sample empirical estimation
start = time.time()
xSamplesLow = sCtg.repeat(samplesPerStimLow,1) + gamma.rsample([int(nSamplesLow)])
# Normalization factor
xSamplesLowNorm = xSamplesLow.norm(dim=1)
# Normalize each random sample
xSamplesLow = torch.einsum('nd,n->nd', xSamplesLow, 1/xSamplesLowNorm)
# Compute the second moment matrix of the samples
lowEmpCov = torch.einsum('nd,nb->db', xSamplesLow, xSamplesLow) * 1/nSamplesLow
end = time.time()
print(f'Low-samples took: {end-start}')


# <markdowncell>
#
# We now compare the results of the empirical approximation and
# the analytically computed second moment. First we visualize
# the three obtained matrices, and a visualization of the
# estimation error for each matrix element. Then we show a
# scatter plot with the reference empirical value in the X
# axis, and the analytic and low-sample values on the Y axis.
# The spread of the low-sample estimation shows the
# approximation error of this method.

# <codecell>
#########################
# VISUALIZE THE THREE COVARIANCE MATRICES, AND
# THE ESTIMATION ERROR OF EACH ELEMENT
#########################
#
# Compute the maximum errors to get a common color scale
maxErr1 = torch.max(refCov - analyticCov)
maxErr2 = torch.max(refCov - lowEmpCov)
maxErr = max((maxErr1, maxErr2))
#
# Plot the matrices
# Analytic second moment matrix
plt.subplot(2,3,1)
plt.imshow(analyticCov)
plt.title(f'Analytic', fontsize=11)
# Diff Reference - Empirical
plt.subplot(2,3,4)
plt.imshow(refCov - analyticCov, cmap='bwr')
plt.clim(-maxErr, maxErr)
plt.title(f'Reference - Analytic', fontsize=11)
#  Empirical reference
plt.subplot(2,3,2)
plt.imshow(refCov)
plt.title(f'Empirical ref', fontsize=11)
#  Empirical low sample
plt.subplot(2,3,3)
plt.imshow(lowEmpCov)
plt.title(f'Empirical {int(nSamplesLow)} samples', fontsize=11)
#  Diff Reference - :ow sample
plt.subplot(2,3,6)
plt.imshow(refCov - lowEmpCov, cmap='bwr')
plt.clim(-maxErr, maxErr)
plt.title(f'Reference - Low sample', fontsize=11)
fig = plt.gcf()
fig.suptitle(f'Second moments d={int(df)}, $\sigma$={pixelSigma}')
plt.setp(fig.get_axes(), xticks=[], yticks=[])
fig.set_size_inches(7,6)
plt.show()

# <codecell>
#########################
# VISUALIZE THE SCATTER PLOT OF EACH ELEMENTS ESTIMATES
# FOR THE TWO ESTIMATION METHODS
#########################
#
# Get the indices of the non-repeated matrix elements (because of symmetry)
nonRepeatedInds = torch.nonzero(torch.tril(torch.ones(df,df)).reshape(df**2))
# Get vectors with the matrix elements for the 3 matrices
smEmpRef = refCov.reshape(df**2)[nonRepeatedInds]
smEmpLow = lowEmpCov.reshape(df**2)[nonRepeatedInds]
smAnalytic = analyticCov.reshape(df**2)[nonRepeatedInds]
# Do scatter plot
plt.scatter(smEmpRef, smEmpLow, c='red', label=f'{nSamplesLow} samples',
        s=1.2, alpha=0.2)
plt.scatter(smEmpRef, smAnalytic, color='blue', label='Analytic',
        s=1.2, alpha=0.2)
# Add identity line
plt.axline((0,0), slope=1, color='black')
plt.show()


# <codecell>
#########################
# COMPUTE AN APPROXIMATION TO THE STIMULUS MEAN, AND THE
# EMPIRICAL REFERENCE TO COMPARE TO
#########################
#
# Compute the mean of the normalized, noisy stimuli
refMean = xSamples.mean(dim=0)
# Use naive normalizing factor, the non-noisy stim inverse norm
normFactor1 = sCtg.norm(dim=1)
sNorm = torch.einsum('nb,n->nb', sCtg, 1/normFactor1)
meanApprox = sNorm.mean(dim=0) # mean of noisy normalized stim
# Use inverse-chi-square normalizing factor
normFactor2 = inv_ncx_batch(mu=sCtg, sigma=pixelSigma)
meanApprox2 = torch.mean(torch.einsum('nb,n->nb', sCtg, normFactor2), dim=0)

plt.plot(meanApprox, color='blue')
plt.plot(meanApprox2, color='red')
plt.plot(refMean, color='black')
plt.show()

ax = au.view_filters_bino_video(refMean.unsqueeze(0),
        frames=nTimesteps, pixels=nPixels)
ax.axes.xaxis.set_visible(True)
ax.axes.yaxis.set_visible(True)
plt.xlabel('Pixels')
plt.ylabel('Time')
plt.title('Left eye                   Right eye')
plt.show()

# Generate the Covariance matrix and torch distribution
#Cov = torch.eye(df)*pixelSigma**2
#gamma = MultivariateNormal(loc=torch.zeros(df), covariance_matrix=Cov)
#analyticCov = torch.zeros(nCtg, df, df)
#analyticDet = torch.zeros(nCtg)
#lowEmpCov = torch.zeros(nCtg, df, df)
#lowEmpDet = torch.zeros(nCtg)
#for ctg in range(nCtg):
#    # Extract the stimuli of this category
#    sCtg = sWeb[torch.nonzero(ctgInd==ctg),:]. squeeze(1)
#    analyticCov[ctg,:,:] = au.isotropic_broadb_sm_batch(sCtg, sigma=pixelSigma)
#    analyticDet[ctg] = torch.linalg.det(analyticCov[ctg,:,:])
#    xSamplesLow = sCtg.repeat(samplesPerStimLow,1) + \
#        gamma.rsample([int(nSamplesLow)]) # for noisy estimation
#    xSamplesLowNorm = xSamplesLow.norm(dim=1) # Normalization factor
#    xSamplesLow = torch.einsum('nd,n->nd', xSamplesLow, 1/xSamplesLowNorm)
#    lowEmpCov[ctg,:,:] = torch.einsum('nd,nb->db', xSamplesLow, xSamplesLow) * \
#        1/nSamplesLow
#    lowEmpDet[ctg] = torch.linalg.det(lowEmpCov[ctg,:,:])
#
#
#ev = torch.real(torch.linalg.eigvals(analyticCov[10,:,:]))
#evm = mpm.matrix(ev)
#
#
#
#plt.plot(analyticDet)
#plt.show()
#
#plt.plot(lowEmpDet)
#plt.show()
#
#
#
#
#ctgPlot = np.arange(1, nCtg, 4)
#nCol = len(ctgPlot)
#for nc in range(nCol):
#    ctg = ctgPlot[nc]
#    plt.subplot(2, nCol, 1+nc)
#    plt.imshow(analyticCov[ctg])
#    plt.subplot(2, nCol, 1+nc+nCol)
#    plt.imshow(lowEmpCov[ctg])
#plt.show()
#
