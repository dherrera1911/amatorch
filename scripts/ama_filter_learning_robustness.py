# <markdowncell>

# #Disparity estimation and reproducibility of learned filters
# 
# Train AMA several times with different seeds, and compare the filters
# learned across runs. Learning is done by filter pairs
# Test the functionality of training the model by training on several
# seeds and selecting the best pair of filters at each run.

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
# ## TEST HOW REPRODUCIBLE THE LEARNED FILTERS ARE, AND TRY DIFFERENT LEARNING PARAMETERS
# 

# <codecell>
##############
#### Set the parameters for training the models
##############

nPairs = 4   # Number of filters to use
filterSigma = float(filterSigmaOri / maxRespOri**2)  # Variance of filter responses
nEpochs = 30
lrGamma = 0.5   # multiplication factor for lr decay
lossFun = cross_entropy_loss()
#lossFun = mse_loss()
learningRate = 0.02
lrStepSize = 10
batchSize = 256

# Put data into Torch data loader tools
trainDataset = TensorDataset(s, ctgInd)
# Batch loading and other utilities 
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)

# Function that returns an optimizer
def opt_fun(model):
    return torch.optim.Adam(model.parameters(), lr=learningRate)
# Function that returns a scheduler
def scheduler_fun(opt):
    return torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize, gamma=lrGamma)

# <codecell>
##############
#### Train an initial model several times, see filter variability
##############

nModels = 5
loss = [None] * nModels
finalLosses = np.zeros((nModels, nPairs))
elapsedTimes = [None] * nModels
filters = [None] * nModels
for n in range(nModels):
    amaPy = AMA(sAll=s, nFilt=2, ctgInd=ctgInd, filterSigma=filterSigma,
        ctgVal=ctgVal)
    loss[n], elapsedTimes[n] = fit_by_pairs(nEpochs=nEpochs, model=amaPy,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt_fun=opt_fun,
        nPairs=nPairs, scheduler_fun=scheduler_fun)
    filters[n] = amaPy.fixed_and_trainable_filters().detach().clone()
    for p in range(nPairs):
        finalLosses[n, p] = loss[n][p][-1]

# Print the loss of the model after each pair of filters is learned.
# Columns indicate the pair of filters, and rows indicate the model instance
print(finalLosses)

# <codecell>
# Plot the learned filters
nFilt = nPairs * 2
for n in range(nModels):
    for nf in range(nFilt):
        plt.subplot(nModels, nFilt, n*nFilt + nf + 1)
        view_filters_bino(filters[n][nf,:])
        plt.yticks([])
        plt.xticks([])
plt.show()


# <codecell>
##############
#### See filter variability when we train several
#### filters at each step and choose the best performing one
##############

nSeeds = 5
nModels = 2
loss = [None] * nModels
finalLosses = np.zeros((nModels, nPairs))
elapsedTimes = [None] * nModels
filters = [None] * nModels

for n in range(nModels):
    amaPy = AMA(sAll=s, nFilt=2, ctgInd=ctgInd, filterSigma=filterSigma,
        ctgVal=ctgVal)
    loss[n], elapsedTimes[n] = fit_by_pairs(nEpochs=nEpochs, model=amaPy,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt_fun=opt_fun,
        nPairs=nPairs, scheduler_fun=scheduler_fun, seedsByPair=nSeeds)
    filters[n] = amaPy.fixed_and_trainable_filters().detach().clone()
    for p in range(nPairs):
        finalLosses[n, p] = loss[n][p][-1]

# Print the loss of the model after each pair of filters is learned.
# Columns indicate the pair of filters, and rows indicate the model instance
print(finalLosses)

# <codecell>
# Plot filters learned by selecting the best filters at each pair
nFilt = nPairs * 2
for n in range(nModels):
    for nf in range(nFilt):
        plt.subplot(nModels, nFilt, n*nFilt + nf + 1)
        view_filters_bino(filters[n][nf,:])
        plt.yticks([])
        plt.xticks([])
plt.show()


# <codecell>
##############
#### Try out different optimization parameters to see if there's
#### differences in the resulting filter variability
##############

nModels = 5
nPairs = 3   # Numbers of pairs of filters to learn
filterSigma = float(filterSigmaOri / maxRespOri**2)  # Variance of filter responses
nEpochs = 40
lrGamma = 0.5   # multiplication factor for lr decay
learningRate = 0.02
lrStepSize = 10

batchSize = [128, 256, 1024]
learningRate = [0.04, 0.01]
lrGamma = [0.8, 0.5]

learnDict = {'batchSize': [], 'learningRate': [], 'lrGamma': [],
        'rep': [], 'filters': [], 'finalLosses': []}
for bs in range(len(batchSize)):
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize[bs], shuffle=True)
    for lr in range(len(learningRate)):
        lrDict = bsDict.copy()
        def opt_fun(model):
            return torch.optim.Adam(model.parameters(), lr=learningRate[lr])
        for g in range(len(lrGamma)):
            gDict = lrDict.copy()
            def scheduler_fun(opt):
                return torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize, gamma=lrGamma[g])
            for n in range(nModels):
                nDict = gDict.copy()
                amaPy = AMA(sAll=s, nFilt=2, ctgInd=ctgInd, filterSigma=filterSigma,
                    ctgVal=ctgVal)
                loss, elapsedTimes = fit_by_pairs(nEpochs=nEpochs, model=amaPy,
                    trainDataLoader=trainDataLoader, lossFun=lossFun, opt_fun=opt_fun,
                    nPairs=nPairs, scheduler_fun=scheduler_fun)
                filters = amaPy.fixed_and_trainable_filters().detach().clone()
                finalLosses = np.zeros(nPairs)
                for p in range(nPairs):
                    finalLosses[p] = loss[p][-1]
                learnDict['batchSize'].append(batchSize[bs])
                learnDict['learningRate'].append(learningRate[lr])
                learnDict['lrGamma'].append(lrGamma[g])
                learnDict['filters'].append(filters)
                learnDict['rep'].append(n)
                learnDict['finalLosses'].append(finalLosses)

learnDict['batchSize'] = np.array(learnDict['batchSize'])
learnDict['learningRate'] = np.array(learnDict['learningRate'])
learnDict['lrGamma'] = np.array(learnDict['lrGamma'])
learnDict['rep'] = np.array(learnDict['rep'])
learnDict['finalLosses'] = np.array(learnDict['finalLosses'])
learnDict['filters'] = np.stack(learnDict['filters'])

# <codecell>
# Make scatter plot with losses of the model filters for different
# parameters
sc = 30
plt.scatter(learnDict['batchSize']+np.random.randint(-sc, sc, 60),
        learnDict['finalLosses'][:,1],
        c=learnDict['learningRate'],
        s=learnDict['lrGamma']**2*100)
plt.colorbar();
plt.show()

# <codecell>
# Plot the learned filters
nFilt = nPairs * 2
inds = np.logical_and.reduce((learnDict['batchSize']==256,
        learnDict['learningRate']==0.04,
        learnDict['lrGamma']==0.5))

filters = learnDict['filters'][inds,:,:]
for n in range(nModels):
    for nf in range(nFilt):
        plt.subplot(nModels, nFilt, n*nFilt + nf + 1)
        view_filters_bino(filters[n,nf,:])
        plt.yticks([])
        plt.xticks([])
plt.show()


