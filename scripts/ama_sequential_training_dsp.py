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

# <codecell>
##############
#### SET TRAINING PARAMETERS FOR FIRST PAIR OF FILTERS
##############
nFilt = 2   # Number of filters to use
filterSigma = float(filterSigmaOri / maxRespOri**2)  # Variance of filter responses
nEpochs = 20
lrGamma = 0.3   # multiplication factor for lr decay
lossFun = nn.CrossEntropyLoss()
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
scheduler = optim.lr_scheduler.StepLR(opt, step_size=lrStepSize,
        gamma=lrGamma)
#opt = torch.optim.SGD(amaPy.parameters(), lr=0.03)  # SGD

# <codecell>
# fit model
loss, elapsedTimes = fit(nEpochs=nEpochs, model=amaPy,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
        scheduler=scheduler)
plt.plot(elapsedTimes, loss)
plt.show()

# <codecell>
## PLOT THE LEARNED FILTERS

plt.subplot(2,2,1)
view_filters_bino(amaPy.f[0,:].detach())
plt.subplot(2,2,2)
view_filters_bino(amaPy.f[1,:].detach())
plt.show()

# Print existing parameters names and size
for name, param in amaPy.named_parameters():
    print(name, param.shape)

# <codecell>
## ADD 2 NEW FILTERS
amaPy.add_new_filters(nFiltNew=2)

# Plot the set of 4 filters before re-training
plt.subplot(2,2,1)
view_filters_bino(amaPy.f[0,:].detach())
plt.subplot(2,2,2)
view_filters_bino(amaPy.f[1,:].detach())
plt.subplot(2,2,3)
view_filters_bino(amaPy.f[2,:].detach())
plt.subplot(2,2,4)
view_filters_bino(amaPy.f[3,:].detach())
plt.show()


# <codecell>
## TRAIN THE NEW FILTERS TOGETHER WITH ORIGINAL
learningRate2 = learningRate * 1/3
nEpochs2 = 20
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
plt.subplot(2,2,1)
view_filters_bino(amaPy.f[0,:].detach())
plt.subplot(2,2,2)
view_filters_bino(amaPy.f[1,:].detach())
plt.subplot(2,2,3)
view_filters_bino(amaPy.f[2,:].detach())
plt.subplot(2,2,4)
view_filters_bino(amaPy.f[3,:].detach())
plt.show()

