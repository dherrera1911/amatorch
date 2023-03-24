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
# **1)** First, because the stimulus is encoded by noisy sensory receptors,
# we add a sample of white noise. Then we normalize to unit lenght.
# $\gamma \in \mathbb{R}^d, \gamma \sim \mathcal{N}\left(0,\sigma_s^2 \right)$
# to the stimulus. Then, we contrast normalize the noisy stimulus:
# \begin{equation}
#   \mathbf{c}_{i,j} = \frac{\mathbf{s}_{i,j}+\mathbf{\gamma}}{\lVert
#       \mathbf{s}_{i,j}+ \mathbf{\gamma} \rVert}
# \end{equation}
#
# **2)** Next, we apply a set of noisy linear filters
# $\mathbf{f} \in \mathbb{R}^{n \times d}$ to
# the contrast-normalized stimulus to obtain a noisy response vector
# $\mathbf{R}_{i,j} \in \mathbb{R}^n$:
#
# \begin{equation}
#   \mathbf{R}_{i,j} = \mathbf{f} \cdot \mathbf{c}_{i,j} + \eta
# \end{equation}
#
# where $\eta \in \mathbb{R}^n, \eta \sim \mathcal{N}\left(0, \sigma_0^2 \right)$
# is a sample of white noise.
#
# **3)** Finally, we obtain the posterior probabilities of each value of the latent
# variable, given the filter responses, $P(X=X_m|\mathbf{R}_{i,j})$. For this we
# compute the likelihood functions
# $L(X=X_m;\mathbf{R}_{i,j}) = P(\mathbf{R}_{i,j}|X_m)$
# and combine them with the class priors $P(X_m)$:
#
# \begin{equation}
#   P(X=X_m|\mathbf{R}_{i,j}) = L(X=X_m; \mathbf{R}_{i,j}) P(X=X_m) $$
# \end{equation}
#
# The details of the probabilistic model to decode the lateng variable
# are explained below.
#
# In the tutorial, we train an AMA model on a set of binocular images to solve
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
##### IMPORT DISPARITY DATA FROM BURGE LAB GITHUB
#!mkdir data
#!wget -O ./data/AMAdataDisparity.mat https://drive.google.com/file/d/17LA4E3F4xrEUnNWOA_jesiSCkDah5asd/view?usp=sharing

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
ctgVal = ctgVal.flatten()


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
# LETS PLOT A RANDOM STIMULUS AND SHOW ITS DISPARITY
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
# First we need to download and install geotorch
#!pip install geotorch
#import geotorch

# <codecell>
# INSTALL THE AMA_LIBRARY PACKAGE FROM GITHUB
#!pip install git+https://github.com/dherrera1911/accuracy_maximization_analysis.git

# <codecell>
# IMPORT THE AMA LIBRARY
import ama_library.ama_class as cl
import ama_library.utilities as au
import ama_library.quadratic_moments as qm


# <markdowncell>
#
# In the ama_library, the ama_class module implements the AMA class, which
# is built on top of the nn.Module from PyTorch. We initialize an AMA
# object to estimate disparity from binocular image patches.
#
# The input parameters to generate the AMA object (mentioned in the
# AMA formulas above):
#
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
# Let's initialize the AMA model:
#

# <codecell>
# Set the parameters
nFilt = 2  # Create the model with 2 filters
pixelNoiseVar = 0.02  # Input pixel noise variance
respNoiseVar = 0.2  # Filter response noise variance
# Create the untrained AMA object
ama = cl.AMA(sAll=s, ctgInd=ctgInd, nFilt=nFilt, respNoiseVar=respNoiseVar,
        pixelCov=pixelNoiseVar, ctgVal=ctgVal)

# <markdowncell>
#
# Let's list some of the basic attributes of the AMA class.
#
# **Attributes:**
# * nFilt: Number of filters in the model ($n$ in the equations)
# * nDim: Number of dimensions in $s$ ($d$ in the equations)
# * nClasses: Number of latent variable levels ($k$ in the equations)
# * f: Filters. Initialized to random variables, these are the trainable
#       parameters, and are constrained to have unit norm.
#       ($\in \mathbb{R}^{n \times d}$).
# * ctgVal: Values of the latent variable ($X_1, X_2, ..., X_k$).
#       ($\in \mathbb{R}^{k}$).
# * stimCov: Covariance matrices for the noisy normalized stimuli
#       $\mathbf{c}$ of each class $X=X_j$.
#       ($\mathbf{\Psi} \in \mathbb{R}^{k \times d \times d}$).
# * stimMean: Means for the noisy-normalized stimuli $\mathbf{c}$ of each class.
#       ($\mathbf{mu} \in \mathbb{R}^{k \times d}$)
# * respCov: Covariance matrices for the noisy responses $\mathbf{R}_{i,j}$
#       (including stimulus variability and filter noise).
#       ($\in \mathbb{R}^{k \times n \times n}$. 
# * respMean: Means for the noisy responses $\mathbf{R}_{i,j}$.
#       ($\in \mathbb{R}^{k \times n}$.
#
# Let's verify that the statistics present in the AMA model match
# what we would expect.

#

# <codecell>
# First visualize randomly initialized filters
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
plt.show()


# <codecell>
# COMPUTE THE COVARIANCE OF THE NOISY NORMALIZED STIMULI FOR CATEGORY j=3
# BY SIMULATION
#
# Load useful functions
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

# Extract the stimuli s for the category
j = 3  # Index of the category to analyze
stimInds = torch.where(ctgInd==j)[0]  # Get indices of the stimuli
sj = s[stimInds, ]  # Extract the stimuli of class j

# ADD NOISE SAMPLES. FOR BETTER POWER, WE ADD MANY NOISE SAMPLES PER STIMULUS
# Make the matrix of pixel covariance noise
pixelNoiseCov = torch.eye(sj.shape[1]) * pixelNoiseVar  # Covariance of stim noise
# Random noise generator
noiseDistr = MultivariateNormal(loc=torch.zeros(sj.shape[1]),
        covariance_matrix=pixelNoiseCov)
# Number of noisy samples to generate for each stimulus
samplesPerStim = 20
# Repeat class stimuli to use many noise samples
repStim = sj.repeat(samplesPerStim, 1)
# Add noise
noisyStim = repStim + noiseDistr.rsample([repStim.shape[0]])
# Normalize to unit norm
noisyNormStim = F.normalize(noisyStim, p=2, dim=1)
# Apply the initialized filters
responses = torch.matmul(noisyNormStim, fInit.transpose(0, 1))

# COMPUTE THE COVARIANCES OF STIMULI AND NOISE
stimCovEmpirical = torch.cov(noisyNormStim.transpose(0,1))
respCovEmpirical = torch.cov(responses.transpose(0,1))

# <codecell>
# Visualize empirically obtained covariances and the ones present in
# the ama object (which are computed analytically)
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
fig.set_size_inches(7,3)
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
# class. Thus, for each class $j$ we have the mean of the class filter
# responses, $\mathbf{mu}_j \in \mathbb{R}^{d}$, and the covariance of
# the responses $\mathbf{\Psi}_j \in \mathbb{R}^{d \times d}$.
#

