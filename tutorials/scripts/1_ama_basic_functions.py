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
# We present the mathematical formalism of the model and the different
# components of AMA class that allow it to solve a task.
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
# 1. Stimulus pre-processing (retinal noise + divisive normalization)
# 1. Linear filtering
# 1. Probabilistic decoding of the latent variable
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
# In this tutorial, we use the AMA model on a set of binocular images to solve
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

# <codecell>
#### DOWNLOAD DISPARITY DATA
##UNCOMMENT_FOR_COLAB_START##
%%capture
!mkdir data
!wget -O ./data/dspCtg.csv https://raw.githubusercontent.com/dherrera1911/accuracy_maximization_analysis/master/data/dspCtg.csv
!wget -O ./data/dspStim.csv https://raw.githubusercontent.com/dherrera1911/accuracy_maximization_analysis/master/data/dspStim.csv
!wget --no-check-certificate -O  ./data/dspVal.csv https://raw.githubusercontent.com/dherrera1911/accuracy_maximization_analysis/master/data/dspVal.csv
!wget --no-check-certificate -O  ./data/dspFilters.csv https://raw.githubusercontent.com/dherrera1911/accuracy_maximization_analysis/master/data/dspFilters.csv
##UNCOMMENT_FOR_COLAB_END##


# <codecell>
##############
#### LOAD DISPARITY DATA
##############
# Load data from csv files
# Load stimuli
s = torch.tensor(np.loadtxt('./data/dspStim.csv', delimiter=','))
s = s.transpose(0,1)
s = s.float()
nPixels = int(s.shape[1]/2)
# Load the category of each stimulus
ctgInd = np.loadtxt('./data/dspCtg.csv', delimiter=',')
ctgInd = torch.tensor(ctgInd, dtype=torch.int64) - 1
# Load the latent variable values
ctgVal = torch.tensor(np.loadtxt('./data/dspVal.csv', delimiter=','))
ctgVal = ctgVal.float()
# Load optimal pre-learned filters
fOpt = torch.tensor(np.loadtxt('./data/dspFilters.csv', delimiter=','))
fOpt = fOpt.float()


# <markdowncell>
# 
# We loaded the binocular images into variable `s`. 
# These stimuli are vertically-averaged images, and so they are
# 1D binocular images (see Burge and Geisler JoV 2014 for details).
# The first half of the columns in `s` contain the left eye image,
# and the second half contain the right eye image.
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
#### PLOT A STIMULUS
##############

# Define function to plot binocular stimulus
def plot_binocular(bino):
    nPixels = int(bino.shape[0]/2)
    x = np.linspace(-30, 30, nPixels)
    # Plot the binocular 1D images
    plt.rcParams.update({'font.size': 14})  # increase default font size
    arcMin = np.linspace(start=-30, stop=30, num=nPixels) # x axis values
    # Plot the binocular 1D images
    plt.plot(x, bino[:nPixels], label='Left eye', color='red')  # plot left eye
    plt.plot(x, bino[nPixels:], label='Right eye', color='blue')  #plot right eye
    plt.xlabel('Visual field (arcmin)')

# Get the disparity value
pltInd = 2011 # index of the stimulus to plot
stimCategoryInd = ctgInd[pltInd]  # select category index (j) for this stim
stimDisparity = ctgVal[stimCategoryInd].numpy()  # Get value of the category (X_j)
# Plot the binocular 1D images
plt.rcParams.update({'font.size': 14})  # increase default font size
arcMin = np.linspace(start=-30, stop=30, num=nPixels) # x axis values
# Plot the binocular 1D images
plot_binocular(s[pltInd,:])
plt.ylabel('Weber contrast')
plt.title(f'Stimulus {pltInd}, Disparity={stimDisparity} arc min disparity')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(7,5)
plt.show()


# <markdowncell>
#
# Next, we will see how the AMA model estimates the latent variable
# (disparity) from the stimulus `s`.


# <markdowncell>
# ## 2) Download AMA library and initialize AMA object
#
# The ama_library can be found in
# https://github.com/dherrera1911/accuracy_maximization_analysis.
# We download and import the library below.

# <codecell>
# FIRST WE NEED TO DOWNLOAD AND INSTALL GEOTORCH AND QUADRATIC RATIOS PACKAGES
##UNCOMMENT_FOR_COLAB_START##
%%capture
!pip install geotorch
import geotorch
##UNCOMMENT_FOR_COLAB_END##

# <codecell>
# INSTALL THE AMA_LIBRARY PACKAGE FROM GITHUB
##UNCOMMENT_FOR_COLAB_START##
%%capture
!pip install git+https://github.com/dherrera1911/accuracy_maximization_analysis.git
##UNCOMMENT_FOR_COLAB_END##

# <codecell>
##############
# IMPORT AMA LIBRARY
##############
import ama_library.ama_class as cl
import ama_library.utilities as au
import ama_library.plotting as ap


# <markdowncell>
#
# The ama_class module implements the AMA class, which
# is built on top of PyTorch's nn.Module. To initialize an AMA
# object we need to specify the following parameters:
#
# * Number of filters: $n$ in the equations, `nFilt` input
# * Pixel noise variance: $\sigma_s^2$ in the equations, `pixelCov` input
# * Response noise variance: $\sigma_0^2$ in the equations, `respNoiseVar` input
#
# We also need to pass the training dataset
# ($\mathbf{s}$, the vector with associated indexes
# $j$, and the latent variable values $X_j$).
# This is in order to compute the stimulus statistics, which are used for
# inference and training. 
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
respNoiseVar = 0.001  # Filter response noise variance
# Create the untrained AMA object
ama = cl.AMA_emp(sAll=s, ctgInd=ctgInd, nFilt=nFilt,
        respNoiseVar=respNoiseVar, pixelCov=pixelNoiseVar, ctgVal=ctgVal)


# <markdowncell>
# The model is initialized with random filters. To better illustrate
# the model, let's change those for pre-trained optimal filters

# <codecell>
# Change random initialization to optimal filters
ama.assign_filter_values(fNew=fOpt)
ama.update_response_statistics()


# <markdowncell>
#
# The filters in the model are in the attribute `f`. Let's plot
# the optimal filters that we put into the model.
#

# <codecell>
##############
# VISUALIZE MODEL FILTERS
##############
fPlot = ama.f.detach().clone()
plt.subplot(1,2,1)
plot_binocular(fPlot[0,:])
plt.ylabel('Weight')
plt.title(f'Filter 1')
plt.ylim(-0.4, 0.4)
plt.subplot(1,2,2)
plot_binocular(fPlot[1,:])
plt.title(f'Filter 2')
plt.ylim(-0.4, 0.4)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(11,5)
plt.show()


# <markdowncell>
# ## 3) Get AMA responses and decode latent variable
#
# Next, we show some basic functionalities of the class, to obtain
# the response to a stimulus and the corresponding latent variable
# estimate.
#
# The AMA model class has different functions for the different processing
# steps mentioned in the introduction to this notebook:
# 1. Stimulus pre-processing (retinal noise + divisive normalization)
# 1. Linear filtering
# 1. Probabilistic decoding of the latent variable


# <markdowncell>
# ### PREPROCESSING
# The function `ama.preprocess()` implements the preprocessing step
# mentioned above. In this case the preprocessing consists of adding
# a sample of noise and normalizing the stimulus to unit norm, but
# other stimulus pre-processing routines can be implemented (e.g.
# narrowband normalization).

# <codecell>
# Let's apply the pre-processing to our dataset
sPre = ama.preprocess(s=s)

# Let's check that the stimulus has unit norm as expected
print(f'Norm of stimulus before preprocessing: {torch.norm(s[0,:])}')
print(f'Norm of stimulus after preprocessing: {torch.norm(sPre[0,:])}')

# <codecell>
# And lets plot one raw stimulus and its pre-processed version
plt.subplot(1,2,1)
plot_binocular(s[pltInd,:])
plt.ylabel('Weber contrast')
plt.title(f'Raw stimulus')
plt.legend()
plt.subplot(1,2,2)
plot_binocular(sPre[pltInd,:])
plt.ylabel('Weber contrast')
plt.title(f'Pre-processed stimulus')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(11,5)
plt.show()


# <markdowncell>
# ### FILTERING
# The method `ama.get_responses()` implements the filtering step,
# and returns the responses of the filters to a stimulus. 
# This function also applies the pre-processing step, so it takes
# as input the raw stimulus s, and returns the filter responses.
# Let's get the responses to the stimuli in the dataset.

# <codecell>
resp = ama.get_responses(s=s).detach()
print(f'Responses shape: {resp.shape}')

# <markdowncell>
# Because we only have 2 filters in the model, we can visualize a
# response as a point in 2D space. Let's plot the responses to
# all stimuli in two different classes. Let's also plot the response
# to the stimulus we plotted above as a black point.

# <codecell>
indPlot1 = ctgInd[pltInd]  # Index of the first class to plot (use same as stim plotted above)
indPlot2 = 12  # Index of the second class to plot
respClass1 = resp[ctgInd==indPlot1, :]  # Get responses of class 1
respClass2 = resp[ctgInd==indPlot2, :]  # Get responses of class 2

plt.scatter(respClass1[:,0], respClass1[:,1], label=f'{ctgVal[indPlot1]} arcmin',
            color='green', alpha=0.5)
plt.scatter(respClass2[:,0], respClass2[:,1], label=f'{ctgVal[indPlot2]} arcmin',
            color='brown', alpha=0.5)
plt.scatter(resp[pltInd,0], resp[pltInd,1], label=f'Plotted stimulus', 
            color='black', s=100)
plt.legend()
plt.xlabel('Filter 1 response')
plt.ylabel('Filter 2 response')
plt.title('Responses to different stimuli')
fig = plt.gcf()
fig.set_size_inches(6,6)
plt.show()


# <markdowncell>
# Note that in the plot above each point shows a different stimulus in the
# dataset. We that the responses to stimuli of the same latent variable
# value (same color) are clustered together into a Gaussian-like cloud of points. 
# We also see that the responses to the two classes are well separated.
# The decoding uses the segregation of responses to decode the latent variable
# (i.e. from the plot above, we can estimate what class the black dot
# corresponds to).
#
# To implement the probabilistic decoding, the model
# approximates the conditional response distributions
# of each class as Gaussian distributions, and uses them to decode the
# latent variable value from the filter responses.


# <markdowncell>
# ### DECODING
# The ama object saves in its attributes the response statistics
# (mean and covariance) conditional on each level of the latent variable.
# These are used to decode the latent variable from the filter responses.
# Lets plot the Gaussians in the attributes `ama.respMean` and
# `ama.respCov` for the responses to the two classes we plotted above.


# <codecell>
# Plot an ellipse of the 95% confidence interval of the response distribution
# for the two classes plotted above.

# First plot the responses like above
ax = plt.subplot(1,1,1)
plt.scatter(respClass1[:,0], respClass1[:,1], label=f'{ctgVal[indPlot1]} arcmin',
            color='green', alpha=0.5)
plt.scatter(respClass2[:,0], respClass2[:,1], label=f'{ctgVal[indPlot2]} arcmin',
            color='brown', alpha=0.5)
plt.legend()
plt.xlabel('Filter 1 response')
plt.ylabel('Filter 2 response')
plt.title('Fitted Gaussians and response scatter')

# Get the response statistics for the two classes
respMean1 = ama.respMean[indPlot1, :].detach()
respCov1 = ama.respCov[indPlot1, :, :].detach()
respMean2 = ama.respMean[indPlot2, :].detach()
respCov2 = ama.respCov[indPlot2, :, :].detach()

# Plot the ellipses of the fitted Gaussians
ap.plot_ellipse(mean=respMean1, cov=respCov1, ax=ax, color='green')
ap.plot_ellipse(mean=respMean2, cov=respCov2, ax=ax, color='brown')
# Show figure
fig = plt.gcf()
fig.set_size_inches(6,6)
plt.show()


# <markdowncell>
# As mentioned above, there are 19 different levels of the latent variable.
# Lets plot the Gaussian ellipses for all the levels of the latent variable.

# <codecell>
# First plot the responses like above
ax = plt.subplot(1,1,1)
plt.xlabel('Filter 1 response')
plt.ylabel('Filter 2 response')
plt.title('Fitted Gaussians')
# Function that plots many ellipses
ap.plot_ellipse_set(mean=ama.respMean, cov=ama.respCov, ax=ax,
                    ctgVal=ctgVal, colorMap='jet')
# Add color legend
ap.add_colorbar(ax=ax, ctgVal=ctgVal, colorMap='jet', label='Disparity \n (arcmin)')
# Set limits
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
# Show figure
fig = plt.gcf()
fig.set_size_inches(9,9)
plt.show()


# <markdowncell>
# We see that similar levels of the latent variable have similar response
# statistics. Also, the responses differ mostly on their second-order
# statistics (i.e. the covariance matrix), and not on their first-order
# statistics (i.e. the mean).


# <markdowncell>
# Approximating the latent-variable-conditional response distributions as
# Gaussian, and using the response mean and covariance of each
# level $j$ ($\mu_j$ and $\Sigma_j$ respectively), the likelihood of
# the latent variable is given by
#
# \begin{equation}
#     L(X=X_j;\mathbf{R}) = P(\mathbf{R} | X=X_j) =
#     \frac{1}{\sqrt{(2\pi)^n |\mathbf{\Psi_j}|}}
#     \exp\left( -\frac{1}{2} (\mathbf{R}-\boldsymbol{\mu}_j)^T
#     \mathbf{\Psi}_j^{-1} (\mathbf{R}-\boldsymbol{\mu}_j) \right)
# \end{equation}
#
# Using Bayes' rule, we can then obtain the posterior probability of
# each class given the filter responses:
#
# \begin{equation}
#   P(X=X_j | \mathbf{R}) = \frac{P(\mathbf{R} | X=X_j) P(X=X_j)}{\sum_{i=1}^{i=k}
#           P(\mathbf{R} | X=X_i) P(X=X_i)}
# \end{equation}
# 
# The AMA class has different methods to obtain the likelihoods and the
# posteriors for a stimulus. The methods `ama.get_ll(s=s)`
# and `ama.get_posteriors(s=s)` return the log-likelihoods and the
# posteriors respectively for a set of stimuli `s`. Alternatively,
# `ama.resp_2_ll(resp=resp)` and `ama.ll_2_posterior(resp=resp)`
# convert the responses into log-likelihoods, and log-likelihoods
# into posteriors respectively.
# We use the former method to obtain the posteriors
# for the stimuli in the dataset.


# <codecell>
# Get the posteriors for the stimuli in the dataset
posteriors = ama.get_posteriors(s=s).detach()
print(f'Posteriors shape: {posteriors.shape}')


# <markdowncell>
# We next plot the posterior probability distribution across all levels of
# the latent variable for all stimuli in the two classes analyzed above.
# We also show the mean posterior for each class, and the true value
# of the latent variable. The different shapes of the posterior
# tell us about the uncertainty of the model about the latent variable.

# <codecell>
# Get the mean posterior for each class
pMean1 = posteriors[ctgInd==indPlot1, :].mean(dim=0)
pMean2 = posteriors[ctgInd==indPlot2, :].mean(dim=0)

# Plot the posteriors for the two classes plotted above
plt.subplot(1,2,1)
plt.plot(ctgVal, posteriors[ctgInd==indPlot1, :].transpose(0,1),
        color='green', alpha=0.2)
plt.plot(ctgVal, pMean1, color='black', linewidth=4)
plt.axvline(x=ctgVal[indPlot1], linewidth=3, linestyle='--', color='black')
plt.xlabel('Disparity (arcmin)')
plt.ylabel('Posterior probability')
plt.title(f'Posterior for {ctgVal[indPlot1]} arcmin')
plt.subplot(1,2,2)
plt.plot(ctgVal, posteriors[ctgInd==indPlot2, :].transpose(0,1),
        color='brown', alpha=0.2)
plt.plot(ctgVal, pMean2, color='black', linewidth=4)
plt.axvline(x=ctgVal[indPlot2], linewidth=3, linestyle='--', color='black')
plt.xlabel('Disparity (arcmin)')
plt.ylabel('Posterior probability')
plt.title(f'Posterior for {ctgVal[indPlot2]} arcmin')
fig = plt.gcf()
fig.set_size_inches(11,5)
plt.show()



# <markdowncell>
# Finally, we can use the posterior distributions to obtain an estimate
# of the latent variable. Like before, the AMA class has two methods to
# obtain estimates, `ama.get_estimates(s=s)` that takes as input a set of
# stimuli, and `ama.posterior_2_estimate(posteriors=posteriors)` to
# convert an array of posteriors to estimates.
# For this, AMA uses the attribute `ctgVal` that was given at initialization.
# The default method for obtaining estimates is the *Maximum A Posteriori*
# (MAP) estimate, which returns the value of the latent variable with
# the highest posterior probability.
#
# Below, we plot the mean estimate for each category of the latent variable


# <codecell>
# Get the estimates for each stimulus
estim = ama.get_estimates(s=s).detach()
print(f'Disparity estimated for 5 first stimuli: {estim[0:5]}')

# <markdowncell>
# Lets plot the mean estimate for each category of the latent variable.

# <codecell>
# Plot the posteriors for the two classes plotted above
estimStats = au.get_estimate_statistics(estimates=estim, ctgInd=ctgInd)
# Plot the mean estimates for each class
fig, ax = plt.subplots()
ax.plot(ctgVal, estimStats['estimateMean'])
plt.fill_between(ctgVal, estimStats['lowCI'], estimStats['highCI'],
        color='blue', alpha=0.2, label='95% CI')
ax.axline((0, 0), slope=1, color='black')
plt.ylim(ctgVal.min(), ctgVal.max())
plt.ylabel('Mean estimated disparity (arcmin)')
plt.xlabel('True disparity (arcmin)')
plt.show()



# <markdowncell>
# ## Overview
# In this notebook we showed the basic functionality of the AMA class,
# which implements the AMA model. The model applies 3 processing steps
# to solve a task using a stimulus: stimulus pre-processing, linear
# filtering, and probabilistic decoding of the latent variable.
# We described methods of the class that return filter responses, probability
# distributions and estimates for an input stimulus. We also showed that
# the model filters are in attribute `ama.f` and that the response
# statistics for each level of the latent variable are in attributes
# `ama.respMean` and `ama.respCov`.
# 
# In the next tutorial we show how the AMA mode is trained to learn
# optimal filters for a given task.


