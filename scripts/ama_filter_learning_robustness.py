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
#from ama_library import *

# <codecell>
#### UNCOMMENT THIS CELL FOR GOOGLE COLAB EXECUTION
!pip install geotorch
import geotorch
!pip install git+https://github.com/dherrera1911/accuracy_maximization_analysis.git
from ama_library import *
!mkdir data
!wget -O ./data/AMAdataDisparity.mat https://github.com/burgelab/AMA/blob/master/AMAdataDisparity.mat?raw=true


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
nEpochs = 50
lrGamma = 0.5   # multiplication factor for lr decay
lossFun = nn.CrossEntropyLoss()
learningRate = 0.02
lrStepSize = 10
batchSize = 1024

# Put data into Torch data loader tools
trainDataset = TensorDataset(s, ctgInd)
# Batch loading and other utilities 
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize,
        shuffle=True)

# Function that returns an optimizer
def opt_fun(model):
    return torch.optim.Adam(model.parameters(), lr=learningRate)
# Function that returns a scheduler
def scheduler_fun(opt):
    return optim.lr_scheduler.StepLR(opt, step_size=lrStepSize, gamma=lrGamma)

# <codecell>
##############
#### Train an initial model several times, see filter variability
##############

nModels = 4
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
##############
#### Train an initial model several times, see filter variability
##############
# Plot the learned filters
nFilt = 8
for n in range(nModels):
    for nf in range(nFilt):
        plt.subplot(nModels, nFilt, n*nFilt + nf + 1)
        view_filters_bino(filters[n][nf,:])
        plt.yticks([])
        plt.xticks([])
plt.show()

