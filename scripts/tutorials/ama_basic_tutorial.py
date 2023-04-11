# <markdowncell>

# # Tutorial of the AMA-Gauss Python package
#
# Accuracy Maximization Analysis (AMA) is a **dimensionality reduction**
# technique that learns a set of optimal linear
# features to solve a **classification task**,
# given a **Bayesian decoder** of the filter responses. AMA-Gauss
# is a variant of AMA that assumes Gaussian conditional distributions
# of the features (we'll refer to AMA-Gauss as AMA throghout).
#
# AMA has been used to train image-computable **ideal observer models**
# for different visual tasks (estimation of retinal speed,
# disparity, 3D motion, defocus).
# Unlike other ideal observer models (i.e. models
# using optimal probabilistic inference to solve a perceptual task),
# AMA is image-computable. That is,
# while most ideal observers receive as input a noisy estimate of
# the latent variable of interest (without specifying how it is
# estimated from the raw input), AMA receives the raw
# high-dimensional image and uses it to estimate the latent variable.
#
# Unlike other models used to learn optimal sensory
# encodings of natural image statistics which use efficient
# coding, or reconstruction error (e.g. sparse coding), AMA learns the
# optimal encoding to solve a specific sensory tasks.
#
# Here we introduce a PyTorch implementation of AMA, trained through
# gradient descent.
# We present the mathematical formalism of the model, the different
# components of AMA class, and the
# functionalities to train and test an AMA model on a set of stimuli.
# As an study case, we train AMA on the task of disparity estimation from
# binocular images.
#
# ## Basic structure of AMA-Gauss
#
# Let $\mathbf{s}_{i,j} \in \mathbb{R}^d$ be an input stimulus
# (e.g. a binocular image) that is the $i^{th}$ stimulus associated
# the true value $X_j$ of the latent variable $X$
# (e.g. the disparity of the image).
# The latent variable can take values (e.g. a given disparity value in
# arc min) from a set ${X_1, X_2, ..., X_k}$.
# The goal of AMA is to compute the
# posterior probability distribution over $X$, which can be used to read out
# an estimate of the latent variable for the input image. This will be made
# clearer below.
#
# The AMA-Gauss model consists of 3 stages:
#
# 1. Noisy encoding of the stimulus and contrast normalization
# 1. Apply noisy linear filters to the noisy normalized stimulus
# 1. Computing the posterior distribution over the latent variable
#       from the filter responses
#
# **1)** Add a sample of white noise
# $\gamma \in \mathbb{R}^d, \gamma \sim \mathcal{N}\left(0,\sigma_s^2 \right)$
# to the stimulus, to simulate noisy sensory receptors.
# Then normalize to unit lenght:
# \begin{equation}
#   \mathbf{c}_{i,j} = \frac{\mathbf{s}_{i,j}+\mathbf{\gamma}}{\lVert
#       \mathbf{s}_{i,j}+ \mathbf{\gamma} \rVert}
# \end{equation}
#
# **2)** Apply a set of noisy linear filters
# $\mathbf{f} \in \mathbb{R}^{n \times d}$ to
# the contrast-normalized stimulus, obtaining a population response vector
# $\mathbf{R}_{i,j} \in \mathbb{R}^n$:
#
# \begin{equation}
#   \mathbf{R}_{i,j} = \mathbf{f} \cdot \mathbf{c}_{i,j} + \eta
# \end{equation}
#
# where $\eta \in \mathbb{R}^n, \eta \sim \mathcal{N}\left(0, \sigma_0^2 \right)$
# is a sample of white noise.
#
# **3)** Given the filter responses, compute the posterior probabilities
# of each value of the latent variable, $P(X=X_m|\mathbf{R}_{i,j})$. For this,
# compute the likelihood functions 
# $L(X=X_m;\mathbf{R}_{i,j}) = P(\mathbf{R}_{i,j}|X_m)$
# (details will be given below), and combine them with
# the class priors $P(X_m)$:
#
# \begin{equation}
#   P(X=X_m|\mathbf{R}_{i,j}) = L(X=X_m; \mathbf{R}_{i,j}) P(X=X_m)
# \end{equation}
#
# In this tutorial, we train an AMA model on a set of binocular images to solve
# the task of estimating disparity.

# <markdowncell>
# ## 1) Import and visualize the data
#
# We first download and import the binocular images and their
# disparity values from the [Burge lab](http://burgelab.psych.upenn.edu/) GitHub page.

# <codecell>
##############
#### IMPORT PACKAGES
##############
import scipy.io as spio
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# <codecell>
#### DOWNLOAD DISPARITY DATA
##UNCOMMENT_FOR_COLAB_START##
#!mkdir data
#!wget -O ./data/ama_dsp_noiseless.mat https://www.dropbox.com/s/eec1917swc124qd/ama_dsp_noiseless.mat?dl=0
##UNCOMMENT_FOR_COLAB_END##

# <codecell>
##############
#### LOAD DISPARITY DATA
##############
# Load ama struct from .mat file into Python
data = spio.loadmat('./data/ama_dsp_noiseless.mat')
# Extract disparity stimuli
s = data.get("s")
s = torch.from_numpy(s)
s = s.transpose(0,1)
s = s.float()
# Get number of pixels
nPixels = int(s.shape[1]/2)
# Extract the vector indicating category of each stimulus row
ctgInd = data.get("ctgInd")
ctgInd = torch.tensor(ctgInd)
ctgInd = ctgInd.flatten()
ctgInd = ctgInd-1       # convert to python indexing (subtract 1)
ctgInd = ctgInd.type(torch.LongTensor)  # convert to torch integer
# Extract the values of the latent variable
ctgVal = data.get("X")
ctgVal = torch.from_numpy(ctgVal)
ctgVal = ctgVal.flatten().float()


# <markdowncell>
# 
# We loaded the binocular images into variable `s`. 
# The stimuli that we loaded are vertically-averaged versions
# of the original stimuli, and thus are 1D images
# (see Burge and Geisler JoV 2014 for details).
#
# We loaded into variable `ctgInd` a vector containing the index $j$ of the
# true level $X_j$ of the latent variable $X$ associated with each stimulus.
# In `ctgVal` we loaded the set of possible levels
# that $X$ can take.
#
# Let's take a look at the data:


# <codecell>
print(f'Image dataset s has {s.shape[0]} images, of {s.shape[1]} pixels each ({nPixels} pixels per monocular image)')
print(f'ctgInd is a vector of length {len(ctgInd)}, with the category index of each s')
print(f'ctgVal is a vector of length {len(ctgVal)} containing the possible values of X')
print(f'ctgVal ranges between {min(ctgVal)} and {max(ctgVal)} arcmin')


# <codecell>
##############
#### PLOT RANDOM STIMULUS
##############
plt.rcParams.update({'font.size': 15})  # increase default font size
arcMin = np.linspace(start=-30, stop=30, num=nPixels) # x axis values
randomInd = np.random.randint(s.shape[0])  # Select a random stimulus
# Get the disparity value
stimCategoryInd = ctgInd[randomInd]  # select category index (j) for this stim
stimDisparity = ctgVal[stimCategoryInd].numpy()  # Get value of the category (X_j)
# Plot the binocular 1D images
x = np.linspace(-30, 30, nPixels)
plt.plot(x, s[randomInd, :nPixels], label='Left', color='red')  # plot left eye
plt.plot(x, s[randomInd, nPixels:], label='Right', color='blue')  #plot right eye
plt.ylabel('Weber contrast')
plt.xlabel('Visual field (arcmin)')
plt.title(f'Stimulus {randomInd}, with j={stimCategoryInd}, $X_j$={stimDisparity} arc min disparity')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(7,5)
plt.show()


# <markdowncell>
#
# We saw above the main inputs that the model will receive:
# the stimuli matrix `s`, their latent variable indices `ctgInd`, and
# the latent variable values `ctgVal`. Next, we'll see the structure of the
# AMA-Gauss model


# <markdowncell>
# ## 2) Download AMA library and initialize AMA object
#
# The ama_library (under development) can be found in
# https://github.com/dherrera1911/accuracy_maximization_analysis.
# We download and import the library below.

# <codecell>
# FIRST WE NEED TO DOWNLOAD AND INSTALL GEOTORCH
##UNCOMMENT_FOR_COLAB_START##
#!pip install geotorch
#import geotorch
##UNCOMMENT_FOR_COLAB_END##

# <codecell>
# INSTALL THE AMA_LIBRARY PACKAGE FROM GITHUB
##UNCOMMENT_FOR_COLAB_START##
#!pip install git+https://github.com/dherrera1911/accuracy_maximization_analysis.git
##UNCOMMENT_FOR_COLAB_END##

# <codecell>
##############
# IMPORT AMA LIBRARY
##############
import ama_library.ama_class as cl
import ama_library.utilities as au


# <markdowncell>
#
# In the ama_library, the ama_class module implements the AMA class, which
# is built on top of the nn.Module from PyTorch. We initialize an AMA
# object to estimate disparity from binocular image patches.
#
# The input parameters to generate the AMA object (mentioned in the
# AMA formulas above):
##
# * Number of filters: $n$ in the equations, `nFilt` input
# * Pixel noise variance: $\sigma_s^2$ in the equations, `pixelCov` input
# * Response noise variance: $\sigma_0^2$ in the equations, `respNoiseVar` input
#
# We need to pass the training dataset as input (the matrix with
# $\mathbf{s}$, the vector with associated indexes
# $j$, and the latent variable values $X_j$).
# This is because the statistics of the stimulus and of filter responses
# are stored in the AMA object. Thus, the training dataset is used to
# estimate the statistics for the initial random filters
#
# Let's initialize the AMA model (details about the inputs to AMA
# initialization can be found in the
# [AMA GitHub](https://github.com/dherrera1911/accuracy_maximization_analysis/blob/master/ama_library/ama_class.py)):
#


# <codecell>
##############
# INITIALIZE AMA MODEL
##############
# Set the parameters
nFilt = 2  # Create the model with 2 filters
pixelNoiseVar = 0.001  # Input pixel noise variance
respNoiseVar = 0.003  # Filter response noise variance
# Create the untrained AMA object
ama = cl.AMA(sAll=s, ctgInd=ctgInd, nFilt=nFilt, respNoiseVar=respNoiseVar,
        pixelCov=pixelNoiseVar, ctgVal=ctgVal,
        respCovPooling='pre-filter', filtNorm='broadband')


# <markdowncell>
#
# Let's list some of the basic attributes of the AMA class.
#
# **Attributes:**
# * `nFilt`: Number of filters in the model ($n$ in the equations)
# * `nDim`: Number of dimensions in $s$ ($d$ in the equations)
# * `nClasses`: Number of latent variable levels ($k$ in the equations)
# * `f`: Filters. Initialized to random variables, these are the trainable parameters, and are constrained to have unit norm. ($\in \mathbb{R}^{n \times d}$).
# * `ctgVal`: Values of the latent variable ($X_1, X_2, ..., X_k$). ($\in \mathbb{R}^{k}$).
# * `stimCov`: Covariance matrices for the noisy normalized stimuli $\mathbf{c}$ of each class $X=X_j$. ($\in \mathbb{R}^{k \times d \times d}$).
# * `stimMean`: Means for the noisy-normalized stimuli $\mathbf{c}$ of each class. ($\in \mathbb{R}^{k \times d}$)
# * `respCov`: Covariance matrices for the noisy responses $\mathbf{R}_{i,j}$ (including stimulus variability and filter noise). ($\in \mathbb{R}^{k \times n \times n}$). 
# * `respMean`: Means for the noisy responses $\mathbf{R}_{i,j}$. ($\in \mathbb{R}^{k \times n}$).
#
# Let's verify that the statistics present in the AMA model match
# what we would expect.


# <codecell>
##############
# VISUALIZE RANDOMLY INITIALIZED FILTERS
##############
fInit = ama.f.detach().clone()
plt.subplot(1,2,1)
plt.plot(x, fInit[0, :nPixels], label='Left', color='red')  # plot left eye
plt.plot(x, fInit[0, nPixels:], label='Right', color='blue')  #plot right eye
plt.ylabel('Weight')
plt.xlabel('Visual field (arcmin)')
plt.title(f'Filter 1, random init')
plt.ylim(-0.4, 0.4)
plt.subplot(1,2,2)
plt.plot(x, fInit[1, :nPixels], label='Left', color='red')  # plot left eye
plt.plot(x, fInit[1, nPixels:], label='Right', color='blue')  #plot right eye
plt.xlabel('Visual field (arcmin)')
plt.title(f'Filter 2, random init')
plt.ylim(-0.4, 0.4)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(11,5)
plt.show()

# <codecell>
##############
# COMPUTE EMPIRICAL COVARIANCES OF NOISY NORMALIZED STIMULI
##############
# Load useful functions
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

# Extract the stimuli s for the category
j = 8  # Index of the category to analyze
stimInds = torch.where(ctgInd==j)[0]  # Get indices of the stimuli
sj = s[stimInds, ]  # Extract the stimuli of class j

# ADD NOISE SAMPLES. FOR BETTER POWER, WE ADD MANY NOISE SAMPLES PER STIMULUS
# Make the matrix of pixel covariance noise
pixelNoiseCov = torch.eye(sj.shape[1]) * pixelNoiseVar  # Covariance of stim noise
# Random noise generator
noiseDistr = MultivariateNormal(loc=torch.zeros(sj.shape[1]),
        covariance_matrix=pixelNoiseCov)
# Number of noisy samples to generate for each stimulus
samplesPerStim = 1000
# Repeat class stimuli to use many noise samples
repStim = sj.repeat(samplesPerStim, 1)
# Add noise
noisyStim = repStim + noiseDistr.rsample([repStim.shape[0]])
# Normalize to unit norm
noisyNormStim = F.normalize(noisyStim, p=2, dim=1)
# Apply the initialized filters
responses = torch.matmul(noisyNormStim, fInit.transpose(0, 1))
# COMPUTE THE COVARIANCES OF NOISY NORMALIZED STIMULI
stimCovEmpirical = torch.cov(noisyNormStim.transpose(0,1))
respCovEmpiricalNoiseless = torch.cov(responses.transpose(0,1))
# Add response noise
respCovEmpirical = respCovEmpiricalNoiseless + torch.eye(nFilt) * respNoiseVar

# <codecell>
##############
# COMPARE EMPIRICAL COVARIANCES AND AMA-ESTIMATED COVARIANCES
##############
plt.subplot(2,2,1)
plt.imshow(stimCovEmpirical)
plt.title('Cov(s) [empirical]')
plt.subplot(2,2,2)
plt.imshow(ama.stimCov[j,:,:].detach())
plt.title('Cov(s) [AMA]')
plt.subplot(2,2,3)
plt.imshow(respCovEmpirical)
plt.title('Cov(R) [empirical]')
plt.subplot(2,2,4)
plt.imshow(ama.respCov[j,:,:].detach())
plt.title('Cov(R) [AMA]')
fig = plt.gcf()
fig.suptitle('Stimulus and response covariance for one class. Empirical vs AMA')
plt.setp(fig.get_axes(), xticks=[], yticks=[])
fig.set_size_inches(10,10)
plt.show()


# <markdowncell>
#
# ## 3) Decoding in AMA model
#
# Now that we layed out the AMA model structure, we next show the decoding
# of the latent class using the AMA object.
#
# To decode the latent variable class, we assume that the noisy filter
# responses are Gaussian distributed, conditional on the latent variable
# class. Thus, for each class $j$ we have the mean of the filter
# responses to the class stimuli, $\mathbf{mu}_j \in \mathbb{R}^{d}$,
# and the covariance of the responses
# $\mathbf{\Psi}_j \in \mathbb{R}^{d \times d}$.
#
# Thus, the probability of observing the response vector $\mathbf{R}_i$
# if $X = X_j$ is given by:
#
# \begin{equation}
#     P(\mathbf{R}_i | X=X_j) = \frac{1}{\sqrt{(2\pi)^n |\mathbf{\Psi_j}|}}
#     \exp\left( -\frac{1}{2} (\mathbf{R}_i-\boldsymbol{\mu}_j)^T
#     \mathbf{\Psi}_j^{-1} (\mathbf{R}_i-\boldsymbol{\mu}_j) \right)
# \end{equation}
#
# As shown above, each $\mathbf{\Psi}_j$ and $\boldsymbol{\mu}$ was computed
# for the initial random filters. The attribute `ama.respCov[j,:,:]` contains
# $\mathbf{\Psi}_j$ and the attribute `ama.respMean[j,:]` contains
# $\boldsymbol{\mu}_j$.
#
# Assuming a flat prior $P(X=X_j) = \frac{1}{k}$, the posterior
# distribution over the latent variable given a filter population
# response $\mathbf{R}_i$ is then given by the following formula:
# 
# \begin{equation}
#   P(X=X_j | \mathbf{R}_i) = \frac{P(\mathbf{R}_i | X=X_j)}{\sum_{i=1}^{i=k}
#           P(\mathbf{R}_i | X=X_i)}
# \end{equation}
# 
# We can then use the posterior distribution to decode a value of $X$,
# for example by choosing the *Minimum Mean Square Estimate* (MMSE) or the
# *Maximum A Posteriori* estimate. This completes the probabilistic
# ideal-observer model that takes stimuli as inputs, and generates
# stimulus estimates as outputs.
# 
# We next show how to carry out this process with the AMA object.


# <markdowncell>
#
# ### Visualize response statistics
#
# Let's first see the distribution of responses in the dataset.
# We will use a function included in the AMA object that computes
# the responses to a set of input stimuli, and plotting functions
# from the `utility` module.

# <codecell>
##############
# PLOT DISTRIBUTION OF NOISY FILTER RESPONSES
##############
au.all_response_ellipses(model=ama, s=s, ctgInd=ctgInd,
        ctgStep=4, colorLabel='Disparity (arcmin)')


# <markdowncell>
#
# ### Visualize stimulus posteriors
#
# The latent-variable conditional Gaussian distributions shown above
# as ellipses are used for decoding the latent variable from filter
# responses. Below, we show the posterior distributions obtained
# for the responses to each stimulus of a given category, using the set
# of distributions shown above. Then, we show the mean estimate
# for the all the stimuli across each category

# <codecell>
##############
# VISUALIZE POSTERIOR DISTRIBUTIONS
##############
j = 6  # category to visualize
jInd = torch.where(ctgInd == j)[0]  # Select stimuli indices
ctgPosteriors = ama.get_posteriors(s=s[jInd,:],
        addStimNoise=True, addRespNoise=True)  # Compute posteriors
ctgPosteriors = ctgPosteriors.detach()  # detach from pytorch gradient
# Plot the posteriors
plt.plot(ctgVal, ctgPosteriors.transpose(0,1), color='black', linewidth=0.2)
plt.axvline(x=ctgVal[j], color='red')
plt.show()

# <codecell>
##############
# VISUALIZE THE mean estimates of the model for each model category
# CATEGORY
##############
# Get the estimates for each stimulus using the current untrained filters
estimUntr = ama.get_estimates(s=s, method4est='MAP', addStimNoise=True,
        addRespNoise=True)
# Summarize the estimates into means and SD for each class
estimSummUntr = au.get_estimate_statistics(estimates=estimUntr,
        ctgInd=ctgInd)
# Plot the mean estimates for each class
fig, ax = plt.subplots()
ax.plot(ctgVal, estimSummUntr['estimateMean'])
plt.fill_between(ctgVal, estimSummUntr['lowCI'], estimSummUntr['highCI'],
        color='blue', alpha=0.2, label='95% CI')
ax.axline((0, 0), slope=1, color='black')
plt.ylim(ctgVal.min(), ctgVal.max())
plt.show()

# <markdowncell>
#
# ## 4) Training the model
#
# So far, we showed how the AMA object takes a stimulus and
# generates a posterior probability distribution over latent-variable
# values. But the model was initialized with random filters, and
# so the behavior of the model is not very useful or informative.
#
# However, all the operations discussed above to get from the stimuli
# to the posteriors are differentiable with respect to the filters.
# This means that we can define a loss function, and use standard
# Pytorch tools to perform gradient descent on the filter to
# minimize the loss. Below, we show the procedure to train the model,
# using functions defined in ama.utilities. We then show the
# results of training the model


# <codecell>
##############
# DEFINE MODEL TRAINING PARAMETERS
##############
nEpochs = 50
lrGamma = 0.3   # multiplication factor for lr decay
lrStepSize = nEpochs/3
learningRate = 0.01
batchSize = 1024
#lossFun = au.nll_loss()  # Negative-log-likelihood loss (uses n-log-likelihood)
lossFun = au.cross_entropy_loss()  # Cross-entropy loss (uses posteriors)

##############
# GENERATE OBJECTS REQUIRED FOR TRAINING
##############
# Put data into Torch data loader tools
trainDataset = TensorDataset(s, ctgInd)
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize,
    shuffle=True)
# Set up optimizer
opt = torch.optim.Adam(ama.parameters(), lr=learningRate)  # Adam
# Set up scheduler to adapt learning rate
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize,
        gamma=lrGamma)


# <codecell>
##############
# TRAIN THE MODEL
##############
loss, time = au.fit(nEpochs=nEpochs, model=ama,
        trainDataLoader=trainDataLoader, lossFun=lossFun,
        opt=opt, sAll=s, ctgInd=ctgInd, scheduler=scheduler)


# <markdowncell>
#
# ### Visualize trained model
#
# Now the model is trained, let's look at what changed. The functions
# below also show how to use the trained model (e.g. to get predictions).
#

# <codecell>
##############
# PLOT LOSS THROUGH TRAINING
##############
plt.plot(loss)
plt.show()


# <codecell>
##############
# PLOT RESULTING FILTERS
##############
au.view_all_filters_1D_bino_image(ama)
plt.show()

# <markdowncell>
#
# We see above that the filters are learned very quickly,
# mostly in only a few epochs. Also, the resulting filters
# look reasonable, forming a pair of filters each of which is
# smooth, symmetric across the two eyes, and symmetric with
# the other filter

# <codecell>
##############
# PLOT RESPONSE DISTRIBUTION WITH TRAINED MODEL
##############
au.all_response_ellipses(model=ama, s=s, ctgInd=ctgInd,
        ctgStep=4, colorLabel='Disparity (arcmin)')
fig.set_size_inches(10,8)


# <markdowncell>
#
# Above we also see that the distribution of filter responses seems
# well approximated by the Gaussian ellipses, indicating that our
# assumption of conditional Gaussian distribution of filter responses
# $\mathbf{R}$ is supported in this case. Also, as could be expected from
# the properties of the filters and the images (both with 0 mean, and
# the images approximately translation invariant), the mean filter
# responses are approximately 0 for each class, and classes are mostly
# separated by their second-order statistics.


# <codecell>
##############
# PLOT THE ESTIMATES MEAN FOR EACH CATEGORY
##############
# Get estimates with the new trained model
estimTrained = ama.get_estimates(s=s, method4est='MAP', addStimNoise=True,
        addRespNoise=True)
estimTrained = estimTrained.detach()
estimSummTrained = au.get_estimate_statistics(estimates=estimTrained,
        ctgInd=ctgInd)
# Plot the estimates means and CIs
fig, ax = plt.subplots()
ax.plot(ctgVal, estimSummTrained['estimateMean'])
plt.fill_between(ctgVal, estimSummUntr['lowCI'], estimSummUntr['highCI'],
        color='blue', alpha=0.2, label='95% CI')
ax.axline((0, 0), slope=1, color='black')
plt.ylim(ctgVal.min(), ctgVal.max())
plt.show()


# <markdowncell>
#
# Above we finally see that the separation of response distributions
# obtained after model training translated into an improvement in model
# performance, although model performance is still far from optimal with
# only 2 filters.


# <markdowncell>
#
# ## 5) Overview
#
# In this notebook we showed both the basic structure of the AMA
# model, and the basic functionalities of the AMA implementation in
# Pytorch.
# 
# Some of the functionalities and behaviors of the model are not
# discussed here. Major functionalities and analyses of the model
# that are (or will be) shown in other notebooks are:
#    - Other filter response normalization regimes
#    - Accuracy-speed tradeoffs in stimulus and response statistics approximation
#    - Procedures for reproducible and interpretable training of many filters 
#    - Training on stimuli with more than 1D (e.g. 2D images, videos)
#    - Geometrical analysis of the latent variable in the manifold of response statistics
#    - Learning of optimal noise correlations, and optimal response normalization
#
