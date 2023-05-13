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
import ama_library.quadratic_moments as qm


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


# Number of models to train with different seeds
nModels = 3
# Models parameters
nPairs = 3   # Number of filters to use
pixelNoiseVar = 0.001  # Input pixel noise variance
respNoiseVar = 0.003  # Filter response noise variance
nEpochs = 20
lrGamma = 0.5   # multiplication factor for lr decay
lossFun = au.cross_entropy_loss()
#lossFun = au.kl_loss()
learningRate = 0.01
lrStepSize = 5
batchSize = 512


#################
# ISOTROPIC
#################

isotropic = cl.Isotropic(sAll=s, ctgInd=ctgInd, nFilt=2,
        respNoiseVar=respNoiseVar, pixelVar=pixelNoiseVar, ctgVal=ctgVal,
        filtNorm='broadband', respCovPooling='pre-filter')

# Put data into Torch data loader tools
trainDataset = TensorDataset(s, ctgInd)
# Batch loading and other utilities 
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize,
        shuffle=True)
# Set up optimizer
opt = torch.optim.Adam(isotropic.parameters(), lr=learningRate)  # Adam
# Make learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize,
        gamma=lrGamma)

# <codecell>
# fit model
loss, elapsedTimes = au.fit(nEpochs=nEpochs, model=isotropic,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
        sAll=s, ctgInd=ctgInd, scheduler=scheduler)
plt.plot(elapsedTimes, loss)
plt.show()

# <codecell>
## PLOT THE LEARNED FILTERS
x = np.linspace(start=-30, stop=30, num=isotropic.nDim) # x axis in arc min
au.view_all_filters_1D_bino_image(isotropic, x=x)
plt.show()


#################
# EMPIRICAL
#################
samplesPerStim = 10
nChannels = 1

empirical = cl.Empirical(sAll=s, ctgInd=ctgInd, nFilt=2,
        respNoiseVar=respNoiseVar, pixelCov=pixelNoiseVar, ctgVal=ctgVal,
        filtNorm='broadband', respCovPooling='pre-filter',
        samplesPerStim=5, nChannels=2)



# Compare empirical and isotrpic stimuli covariances
ind = 8
plt.subplot(1,2,1)
plt.imshow(isotropic.stimCov[ind,:,:])
plt.subplot(1,2,2)
plt.imshow(empirical.stimCov[ind,:,:])
plt.show()

ax = plt.subplot(1,1,1)
plt.scatter(empirical.stimCov[ind,:,:].flatten(),
        isotropic.stimCov[ind,:,:].flatten())
ax.axline((0,0), slope=1, color='black')
plt.show()


# Fit empirical ama

# To fit empirical AMA, we need to generate a noisy, normalized
# stimuli dataset
#sNoisy, ctgIndNoisy = empirical.make_noisy_normalized_stimuli(s=s,
#        ctgInd=ctgInd, samplesPerStim=samplesPerStim)

# Put data into Torch data loader tools
trainDataset = TensorDataset(s, ctgInd)
# Batch loading and other utilities 
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize,
        shuffle=True)
# Set up optimizer
opt = torch.optim.Adam(empirical.parameters(), lr=learningRate)  # Adam
# Make learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize,
        gamma=lrGamma)


loss, elapsedTimes = au.fit(nEpochs=nEpochs, model=empirical,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
        sAll=s, ctgInd=ctgInd, scheduler=scheduler)
plt.plot(elapsedTimes, loss)
plt.show()

# <codecell>
## PLOT THE LEARNED FILTERS
x = np.linspace(start=-30, stop=30, num=empirical.nDim) # x axis in arc min
au.view_all_filters_1D_bino_image(empirical, x=x)
plt.show()


