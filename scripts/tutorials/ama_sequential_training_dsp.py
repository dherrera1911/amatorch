# <markdowncell>

# #Disparity estimation and filter training in pairs
# 
# Train AMA on the task of disparity estimation. Train two
# pairs of filters, one after the other (first the model
# with 2 filters, and then the model with 4 filters)

# <codecell>
##############
#### IMPORT PACKAGES
##############
import scipy.io as spio
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# <codecell>
##UNCOMMENT_FOR_COLAB_START##
#!pip install geotorch
#import geotorch
#!pip install git+https://github.com/dherrera1911/accuracy_maximization_analysis.git
#from ama_library import *
#!mkdir data
#!wget -O ./data/ama_dsp_noiseless.mat https://www.dropbox.com/s/eec1917swc124qd/ama_dsp_noiseless.mat?dl=0
##UNCOMMENT_FOR_COLAB_END##


# <codecell>
import ama_library.ama_class as cl
import ama_library.utilities as au
import ama_library.plotting as ap


# <codecell>
##############
#### LOAD AMA DATA
##############
# Load ama struct from .mat file into Python
data = spio.loadmat('./data/ama_dsp_noiseless.mat')
# Extract contrast normalized, noisy stimulus
s = data.get("s")
s = torch.from_numpy(s)
s = s.transpose(0,1)
s = s.float()
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
nPixels = int(s.shape[1]/2)


# <markdowncell>
# ## TRAINING 2 PAIRS OF FILTERS WITHOUT FIXING ANY
# 
# In this part of the code, we train the model with 2 filters,
# then add 2 new random filters, and continue training the
# 4 filters together. We want to see whether the first 2 filters
# remain fixed through the second round of training.

# <codecell>
##############
#### SET TRAINING PARAMETERS FOR FIRST PAIR OF FILTERS
##############
nFilt = 2   # Number of filters to use
pixelNoiseVar = 0.001  # Input pixel noise variance
respNoiseVar = 0.003  # Filter response noise variance
nEpochs = 30
lrGamma = 0.5   # multiplication factor for lr decay
lossFun = au.cross_entropy_loss()
#lossFun = au.kl_loss()
learningRate = 0.01
lrStepSize = 10
batchSize = 256

# <codecell>
##############
####  TRAIN FIRST PAIR OF FILTERS
##############
# Define model
ama = cl.Isotropic(sAll=s, ctgInd=ctgInd, nFilt=nFilt,
        respNoiseVar=respNoiseVar, pixelVar=pixelNoiseVar, ctgVal=ctgVal, 
        filtNorm='broadband', respCovPooling='pre-filter')

# Put data into Torch data loader tools
trainDataset = TensorDataset(s, ctgInd)
# Batch loading and other utilities 
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize,
        shuffle=True)
# Set up optimizer
opt = torch.optim.Adam(ama.parameters(), lr=learningRate)  # Adam
# Make learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize,
        gamma=lrGamma)

# <codecell>
# fit model
loss, elapsedTimes = au.fit(nEpochs=nEpochs, model=ama,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
        sAll=s, ctgInd=ctgInd, scheduler=scheduler)
plt.plot(elapsedTimes, loss)
plt.show()

# <codecell>
## PLOT THE LEARNED FILTERS
x = np.linspace(start=-30, stop=30, num=ama.nDim) # x axis in arc min
ap.view_all_filters_1D_bino_image(ama, x=x)
plt.show()

# <codecell>
## ADD 2 NEW FILTERS
ama.add_new_filters(nFiltNew=2, sAll=s, ctgInd=ctgInd)

# Plot the set of 4 filters before re-training
ap.view_all_filters_1D_bino_image(ama, x=x)
plt.show()

# <codecell>
## TRAIN THE NEW FILTERS TOGETHER WITH ORIGINAL
learningRate2 = learningRate * 1/3
nEpochs2 = 30
# Re-initializing the optimizer after adding filters is required
opt = torch.optim.Adam(ama.parameters(), lr=learningRate2)  # Adam
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize,
        gamma=lrGamma)
loss, elapsedTimes = au.fit(nEpochs=nEpochs2, model=ama,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
        sAll=s, ctgInd=ctgInd, scheduler=scheduler)
plt.plot(elapsedTimes, loss)
plt.show()

# Plot filters after learning
ap.view_all_filters_1D_bino_image(ama, x=x)
plt.show()

# <markdowncell>
# ## TRAINING 2 PAIRS OF FILTERS, FIXING THE FIRST PAIR
# 
# In this part of the code, we train the model with 2 filters,
# fix these filters so that they are no longer trainable, add
# 2 more filters, and then train these 2 new filters on top of
# the original fixed ones. We aim to see how this procedure compares
# to the training of different filters without fixing.

# <codecell>
# DEFINE NEW MODEL TO TRAIN
ama2 = cl.Isotropic(sAll=s, ctgInd=ctgInd, nFilt=nFilt,
        respNoiseVar=respNoiseVar, pixelVar=pixelNoiseVar, ctgVal=ctgVal, 
        filtNorm='broadband', respCovPooling='pre-filter')

# <codecell>
# SET PARAMETERS FOR TRAINING THE FILTERS. INITIALIZE OPTIMIZER
nEpochs = 40
lrGamma = 0.5   # multiplication factor for lr decay
learningRate = 0.01
lrStepSize = 10
batchSize = 256
# Set up optimizer
opt = torch.optim.Adam(ama2.parameters(), lr=learningRate)  # Adam
# Make learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize,
        gamma=lrGamma)

# <codecell>
# FIT MODEL
loss, elapsedTimes = au.fit(nEpochs=nEpochs, model=ama2,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
        sAll=s, ctgInd=ctgInd, scheduler=scheduler)
plt.plot(elapsedTimes, loss)
plt.show()

ap.view_all_filters_1D_bino_image(ama2, x)
plt.show()

# <codecell>
# ADD FIXED FILTERS
# Fix the learned filters in place
ama2.move_trainable_2_fixed(sAll=s, ctgInd=ctgInd)
# View current filters
ap.view_all_filters_1D_bino_image(ama2, x)
plt.show()

# <codecell>
# TRAIN THE NEW FILTERS WITH THE OLD FILTERS FIXED IN PLACE
# Set up optimizer
opt = torch.optim.Adam(ama2.parameters(), lr=learningRate)  # Adam
# Make learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize,
        gamma=lrGamma)
# fit model
loss, elapsedTimes = au.fit(nEpochs=nEpochs, model=ama2,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
        sAll=s, ctgInd=ctgInd, scheduler=scheduler)
plt.plot(elapsedTimes, loss)
plt.show()
ap.view_all_filters_1D_bino_image(ama2)
plt.show()


# <codecell>
# USE FUNCTION THAT TRAINS AMA FILTERS BY PAIRS
nPairs = 4
# We need to define a function that returns optimizers, because
# a new optimizer has to be generated each time we manually change
# the model parameters
def opt_fun(model):
    return torch.optim.Adam(model.parameters(), lr=learningRate)
# We need to define a function that returns schedulers, because a
# new one has to be defined for each new optimizer
def scheduler_fun(opt):
    return torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize, gamma=lrGamma)

# Initialize model to train
ama3 = cl.Isotropic(sAll=s, ctgInd=ctgInd, nFilt=nFilt,
        respNoiseVar=respNoiseVar, pixelVar=pixelNoiseVar, ctgVal=ctgVal, 
        filtNorm='broadband', respCovPooling='pre-filter')

# <codecell>
# Train model by pairs
loss3, elapsedTimes3 = au.fit_by_pairs(nEpochs=nEpochs, model=ama3,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt_fun=opt_fun,
        nPairs=nPairs, sAll=s, ctgInd=ctgInd, scheduler_fun=scheduler_fun)

# <codecell>
# Visualize trained filters
ap.view_all_filters_1D_bino_image(ama3)
plt.show()

# <codecell>
# View the training loss curves for the learned filters
for l in range(nPairs):
    plt.subplot(1, nPairs, l+1)
    plt.plot(elapsedTimes3[l], loss3[l])
plt.show()

