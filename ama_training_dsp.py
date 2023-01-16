import scipy.io as spio
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import geotorch
from ama_library.ama_class import *
from ama_library.ama_utilities import *
#%%

###### LOAD AMA DATA
# Load ama struct from .mat file into Python
data = spio.loadmat('./data/amaR01_TestData_Disparity.mat')
# Extract contrast normalized, noisy stimulus
s = data.get("sBig")
s = torch.from_numpy(s)
s = s.transpose(0,1)
s = s.float()
# Extract the vector indicating category of each stimulus row
ctgInd = data.get("ctgIndBig")
ctgInd = torch.Tensor(ctgInd)
ctgInd = ctgInd.flatten()
ctgInd = ctgInd-1       # convert to python indexing (subtract 1)
ctgInd = ctgInd.type(torch.LongTensor)  # convert to torch integer
# Extract the values of the latent variable
ctgVal = data.get("X")
ctgVal = torch.from_numpy(ctgVal)
ctgVal = ctgVal.flatten()
nPixels = int(s.shape[1]/2)
#%%

##############
#### Set parameters of the model to train
##############
nFilt = 2 # Number of filters to use
filterSigma = 0.007 # Variance of filter responses 
nEpochs = 30

# Choose loss function
lossFun = nn.CrossEntropyLoss()

##############
#### Analyze effect of batch size
##############
nStim = s.shape[0]
batchFractions = np.array([1/50, 1/20, 1/10, 1/5, 1])
nBatchSizes = batchFractions.size
batchSizeVec = (nStim*batchFractions).astype(int)
learningRateBase = 0.005

filterDict = {"batchSize": [], "filter": [], "loss": [], 'time': [],
        'estimates': [], 'estimateStats': []}

for bs in range(nBatchSizes):
    learningRate = learningRateBase * 10 * batchFractions[bs]
    batchSize = int(batchSizeVec[bs])
    # Initialize model with random filters
    amaPy = AMA(sAll=s, ctgInd=ctgInd, nFilt=nFilt, filterSigma=filterSigma,
            ctgVal=ctgVal)
    # Add norm 1 constraint (set parameters f to lay on a sphere)
    geotorch.sphere(amaPy, "f")
    # Put data into Torch data loader tools
    trainDataset = TensorDataset(s, ctgInd)
    # Batch loading and other utilities 
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize,
            shuffle=True)
    # Set up optimizer
    opt = torch.optim.Adam(amaPy.parameters(), lr=learningRate)  # Adam
    #opt = torch.optim.SGD(amaPy.parameters(), lr=0.03)  # SGD
    # fit model
    loss, elapsedTimes = fit(nEpochs=nEpochs, model=amaPy,
            trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt)
    # Store ama information into dictionary
    filterDict["batchSize"].append(batchSize)
    filterDict["filter"].append(amaPy.f.detach())
    filterDict["loss"].append(loss)
    filterDict["time"].append(elapsedTimes)
    filterDict["estimates"].append(amaPy.get_estimates(s))
    filterDict["estimateStats"].append(
            get_estimate_statistics(filterDict["estimates"][bs], ctgInd))

def view_filters_bino(f, x=[], title=''):
    plt.title(title)
    nPixels = int(max(f.shape)/2)
    if len(x) == 0:
        x = np.arange(nPixels)
    plt.plot(x, f[:nPixels], label='L', color='red')
    plt.plot(x, f[nPixels:], label='R', color='blue')
    plt.ylim(-0.3, 0.3)

x = np.linspace(start=-30, stop=30, num=nPixels) # x axis in arc min
for bs in range(nBatchSizes):
    plt.subplot(2, nBatchSizes, bs+1)
    f1 = filterDict["filter"][bs][0,:]
    view_filters_bino(f=f1, x=x, title='size: %i'  %batchSizeVec[bs])
    plt.subplot(2, nBatchSizes, bs+1+nBatchSizes)
    f2 = filterDict["filter"][bs][1,:]
    view_filters_bino(f=f2, x=x, title=''  %batchSizeVec[bs])
plt.show()

for bs in range(nBatchSizes):
    plt.subplot(2, nBatchSizes, bs+1)
    loss = filterDict["loss"][bs]
    time = filterDict["time"][bs]
    plt.plot(time, loss)
    plt.ylim(2.84, 2.98)
    plt.xlim(0, 3)
    plt.ylabel('Cross entropy loss')
    plt.xlabel('Time (s)')
    plt.title('size: %i'  %batchSizeVec[bs])
    if bs>0:
        plt.yticks([])
        plt.ylabel('')
    plt.subplot(2, nBatchSizes, bs+1+nBatchSizes)
    loss = filterDict["loss"][bs]
    epoch = np.arange(loss.size)
    plt.plot(epoch, loss)
    plt.ylim(2.84, 2.98)
    plt.ylabel('Cross entropy loss')
    plt.xlabel('Epoch')
    if bs>0:
        plt.yticks([])
        plt.ylabel('')
plt.show()

