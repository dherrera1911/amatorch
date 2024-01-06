# <markdowncell>

# # Training the AMA model
#
# In the previous tutorial we showed the AMA functionalities that
# allow to estimate the value of a latent variable from a stimulus.
# In this tutorial we show how to train the model to learn the optimal
# filters for the task.
#
# First we load the data and initialize the AMA model. See the
# previous tutorial for details on the data and the model initialization.


# <codecell>
#### DOWNLOAD DISPARITY DATA
##UNCOMMENT_FOR_COLAB_START##
#!mkdir data
#!wget -O ./data/dspCtg.csv https://raw.githubusercontent.com/dherrera1911/accuracy_maximization_analysis/master/data/dspCtg.csv
#!wget -O ./data/dspStim.csv https://raw.githubusercontent.com/dherrera1911/accuracy_maximization_analysis/master/data/dspStim.csv
#!wget --no-check-certificate -O  ./data/dspVal.csv https://raw.githubusercontent.com/dherrera1911/accuracy_maximization_analysis/master/data/dspVal.csv
#!wget --no-check-certificate -O  ./data/dspFilters.csv https://raw.githubusercontent.com/dherrera1911/accuracy_maximization_analysis/master/data/dspFilters.csv
##UNCOMMENT_FOR_COLAB_END##


# <codecell>
# FIRST WE NEED TO DOWNLOAD AND INSTALL GEOTORCH AND QUADRATIC RATIOS PACKAGES
##UNCOMMENT_FOR_COLAB_START##
#!pip install geotorch
#import geotorch
#!pip install git+https://github.com/dherrera1911/quadratic_ratios.git
##UNCOMMENT_FOR_COLAB_END##


# <codecell>
# INSTALL THE AMA_LIBRARY PACKAGE FROM GITHUB
##UNCOMMENT_FOR_COLAB_START##
#!pip install git+https://github.com/dherrera1911/accuracy_maximization_analysis.git
##UNCOMMENT_FOR_COLAB_END##


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
import ama_library.ama_class as cl
import ama_library.utilities as au
import ama_library.plotting as ap


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


# <codecell>
##############
# INITIALIZE AMA MODEL
##############
# Set the parameters
nFilt = 2  # Create the model with 2 filters
pixelNoiseVar = 0.003  # Input pixel noise variance
respNoiseVar = 0.005  # Filter response noise variance
# Create the untrained AMA object
ama = cl.AMA_emp(sAll=s, ctgInd=ctgInd, nFilt=nFilt,
        respNoiseVar=respNoiseVar, pixelCov=pixelNoiseVar, ctgVal=ctgVal)


# <markdowncell>
# The model is initialized with random filters. Let's visualize the
# random filters.
# the optimal filters that we put into the model.
#

# <codecell>
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
#
# In the previous tutorial we showed that the learned filters separated
# the classes response distributions. However, the untrained filters
# are not expected to do this. Lets visualize the response distributions
# for two classes:

# <codecell>
resp = ama.get_responses(s=s).detach()

indPlot1 = 5 # Index of the first class to plot (use same as stim plotted above)
indPlot2 = 12  # Index of the second class to plot
respClass1 = resp[ctgInd==indPlot1, :]  # Get responses of class 1
respClass2 = resp[ctgInd==indPlot2, :]  # Get responses of class 2
plt.scatter(respClass1[:,0], respClass1[:,1], label=f'{ctgVal[indPlot1]} arcmin',
            color='green', alpha=0.5)
plt.scatter(respClass2[:,0], respClass2[:,1], label=f'{ctgVal[indPlot2]} arcmin',
            color='brown', alpha=0.5)
plt.legend()
plt.xlabel('Filter 1 response')
plt.ylabel('Filter 2 response')
plt.title('Responses to different stimuli')
fig = plt.gcf()
fig.set_size_inches(6,6)
plt.show()


# <markdowncell>
# To learn the filters that maximize performance at the task we perform
# gradient descent, using the Pytorch tools for automatic differentiation.
# We can use different loss functions. In this case we will use a
# cross-entropy loss, that is, the negative log-posterior of the correct
# class:
# 
# \begin{equation}
#     L(R) = -\log P(X=X_j | \mathbf{R})
# \end{equation}
# 
# Let's show how to compute the loss and take a step in the gradient
# direction.
#

# <codecell>
# Initialize Pytorch optimizator for ama parameters
learningRate = 1
opt = torch.optim.Adam(ama.parameters(), lr=learningRate)
opt.zero_grad() # Make sure gradients are zeroed
# Compute the loss for each stimulus
posteriors = ama.get_posteriors(s) # Get posteriors
nStim = s.shape[0]
loss = -torch.log(posteriors[torch.arange(nStim), ctgInd]) # select the correct class
# Print the loss, see that the gradient is kept
print(loss)


# <codecell>
# Now take the mean of the losses, to have a unique loss value
lossMean = loss.mean()
print(f'Initial loss: {lossMean.detach()}')
# Compute the gradient of the loss with respect to the model parameters
lossMean.backward()
# Take a step in the gradient direction
opt.step()


# <markdowncell>
# Now, remember that a key part of the AMA model is that we use the
# filter response statistics to decode the latent variable. Since we
# modified the filters with gradient descent we need to recompute the
# response statistics. This is done with the function
# `ama.update_response_statistics()`. This function makes use of the
# pre-processed stimulus statistics, that don't change with the filters
# (since pre-processing doesn't depend on the filters), and that are
# stored in the attributes `ama.stimMean` and `ama.stimCov`. Basic
# probability theory shows that the mean and covariance of a
# linearly transformed random variable $Y = f^TX$ (i.e. filter outputs) 
# are given by $\mu_{Y} = f^T \mu_X$ and $\Sigma_Y = f^T \Sigma_X f$.
#
# When we create the AMA object, the pre-processed stimulus statistics
# are computed and stored. For the version of AMA implemented here
# (i.e. the empirical version), the pre-processed stimulus statistics
# are computed by taking each stimulus, applying the pre-processing
# (with many samples, since pre-processing is stochastic), and then
# taking the mean and covariance of the pre-processed stimuli.
# Thus, we do not need to pass the stimuli again for computing the
# response statistics.
#
# We update the response statistics, and then compute the loss again
# to see if it decreased.


# <codecell>
ama.update_response_statistics()
# Compute the loss for each stimulus
posteriors = ama.get_posteriors(s) # Get posteriors
nStim = s.shape[0]
loss = -torch.log(posteriors[torch.arange(nStim), ctgInd]) # select the correct class
lossMean = loss.mean()  # Take the mean of the losses
# Print the loss, see that the gradient is kept
print(f'Loss after step: {lossMean.detach()}')


# <markdowncell>
# We see that the loss decreased as expected. The utility functions in the
# AMA package include a basic fitting function that performs the
# gradient descent procedure for a number of epochs, including the
# statistics updating. We will use this function to train the model.
# We also use the loss functions included in the package.

# <codecell>
nEpochs = 30
learningRate = 0.05
lrStepSize = 5 # number of epochs between each lr decay
lrGamma = 0.7   # multiplication factor for lr decay
batchSize = 256

# Put data into Torch data loader tools
trainDataset = TensorDataset(s, ctgInd)
# Batch loading and other utilities 
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
# Optimizer and scheduler
opt = torch.optim.Adam(ama.parameters(), lr=learningRate)
sch = torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize, gamma=lrGamma)

# Fit model
loss, _, elapsedTimes = au.fit(nEpochs=nEpochs, model=ama,
                            trainDataLoader=trainDataLoader,
                            lossFun=au.cross_entropy_loss, opt=opt,
                            scheduler=sch)


# <markdowncell>
# Let's visualize the new filters and how the loss function changed

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

# <codecell>
##############
# VISUALIZE LOSS
##############
plt.plot(torch.arange(nEpochs+1), loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


