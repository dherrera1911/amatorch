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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# <codecell>
##### COMMENT THIS CELL WHEN USING GOOGLE COLAB
from ama_library import *

# <codecell>
#### UNCOMMENT THIS CELL FOR GOOGLE COLAB EXECUTION
#!pip install geotorch
#import geotorch
#!pip install git+https://github.com/dherrera1911/accuracy_maximization_analysis.git
#from ama_library import *
#!mkdir data
#!wget -O ./data/AMAdataDisparity.mat https://github.com/burgelab/AMA/blob/master/AMAdataDisparity.mat?raw=true

# <codecell>
##############
#### LOAD AMA DATA
##############
# Load ama struct from .mat file into Python
data = spio.loadmat('./data/AMAdataDisparity.mat')
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
ctgVal = ctgVal.flatten()
nPixels = int(s.shape[1]/2)
# Extract Matlab trained filters
fOri = data.get("f")
fOri = torch.from_numpy(fOri)
fOri = fOri.transpose(0,1)
fOri = fOri.float()
# Extract original noise parameters
filterSigmaOri = data.get("var0").flatten()
maxRespOri = data.get("rMax").flatten()

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
filterSigma = float(filterSigmaOri / maxRespOri**2)  # Variance of filter responses
nEpochs = 20
lrGamma = 0.5   # multiplication factor for lr decay
#lossFun = nn.CrossEntropyLoss()
lossFun = cross_entropy_loss()
learningRate = 0.01
lrStepSize = 10
batchSize = 256

# <codecell>
##############
####  TRAIN FIRST PAIR OF FILTERS
##############

# Define model
amaPy = AMA(sAll=s, nFilt=nFilt, ctgInd=ctgInd, filterSigma=filterSigma,
        ctgVal=ctgVal)

# Put data into Torch data loader tools
trainDataset = TensorDataset(s, ctgInd)
# Batch loading and other utilities 
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize,
        shuffle=True)
# Set up optimizer
opt = torch.optim.Adam(amaPy.parameters(), lr=learningRate)  # Adam
# Make learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize, gamma=lrGamma)

# <codecell>
# fit model
loss, elapsedTimes = fit(nEpochs=nEpochs, model=amaPy,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
        scheduler=scheduler)
plt.plot(elapsedTimes, loss)
plt.show()

# <codecell>
## PLOT THE LEARNED FILTERS
x = np.linspace(start=-30, stop=30, num=amaPy.nDim) # x axis in arc min
view_all_filters_bino(amaPy, x=x)
#plt.show()

# <codecell>
## ADD 2 NEW FILTERS
amaPy.add_new_filters(nFiltNew=2)

# Plot the set of 4 filters before re-training
view_all_filters_bino(amaPy, x=x)
plt.show()

# <codecell>
## TRAIN THE NEW FILTERS TOGETHER WITH ORIGINAL
learningRate2 = learningRate * 1/3
nEpochs2 = 30
# Re-initializing the optimizer after adding filters is required
opt = torch.optim.Adam(amaPy.parameters(), lr=learningRate2)  # Adam
scheduler = optim.lr_scheduler.StepLR(opt, step_size=lrStepSize,
        gamma=lrGamma)
loss, elapsedTimes = fit(nEpochs=nEpochs2, model=amaPy,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
        scheduler=scheduler)
plt.plot(elapsedTimes, loss)
plt.show()

# Plot filters after learning
view_all_filters_bino(amaPy, x=x)
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
amaPy2 = AMA(sAll=s, nFilt=nFilt, ctgInd=ctgInd, filterSigma=filterSigma,
        ctgVal=ctgVal)

# <codecell>
# SET PARAMETERS FOR TRAINING THE FILTERS. INITIALIZE OPTIMIZER
nEpochs = 40
lrGamma = 0.5   # multiplication factor for lr decay
learningRate = 0.01
lrStepSize = 10
batchSize = 256
# Set up optimizer
opt = torch.optim.Adam(amaPy2.parameters(), lr=learningRate)  # Adam
# Make learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(opt, step_size=lrStepSize,
        gamma=lrGamma)

# <codecell>
# FIT MODEL
loss, elapsedTimes = fit(nEpochs=nEpochs, model=amaPy2,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
        scheduler=scheduler)
plt.plot(elapsedTimes, loss)
plt.show()

view_all_filters_bino(amaPy2, x)
plt.show()

# <codecell>
# ADD FIXED FILTERS
# Fix the learned filters in place
amaPy2.add_fixed_filters(amaPy2.f.detach().clone())
# Re-initialize trainable filters
amaPy2.reinitialize_trainable()
# View current filters
view_all_filters_bino(amaPy2, x)
plt.show()

# <codecell>
# TRAIN THE NEW FILTERS WITH THE OLD FILTERS FIXED IN PLACE
# Set up optimizer
opt = torch.optim.Adam(amaPy2.parameters(), lr=learningRate)  # Adam
# Make learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(opt, step_size=lrStepSize,
        gamma=lrGamma)
# fit model
loss, elapsedTimes = fit(nEpochs=nEpochs, model=amaPy2,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
        scheduler=scheduler)
plt.plot(elapsedTimes, loss)
plt.show()
view_all_filters_bino(amaPy2)
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
    return optim.lr_scheduler.StepLR(opt, step_size=lrStepSize, gamma=lrGamma)

# Initialize model to train
amaPy3 = AMA(sAll=s, nFilt=nFilt, ctgInd=ctgInd, filterSigma=filterSigma,
        ctgVal=ctgVal)

# <codecell>
# Train model by pairs
loss3, elapsedTimes3 = fit_by_pairs(nEpochs=nEpochs, model=amaPy3,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt_fun=opt_fun,
        nPairs=nPairs, scheduler_fun=scheduler_fun)

# <codecell>
# Visualize trained filters
view_all_filters_bino(amaPy3)
plt.show()

# <codecell>
# View the training loss curves for the learned filters
for l in range(nPairs):
    plt.subplot(1, nPairs, l+1)
    plt.plot(elapsedTimes3[l], loss3[l])
plt.show()

# <codecell>
# Visualize MATLAB AMA filters
for n in range(fOri.shape[0]):
    plt.subplot(2,2,n+1)
    view_filters_bino(fOri[n,:])
plt.show()

