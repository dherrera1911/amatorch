# <markdowncell>

# #Disparity estimation and effect of batch size
# 
# Train AMA on the task of disparity estimation. Compare
# the filters learned with different batch sizes, as well
# as model performance and training time

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
import geotorch
from ama_library import *

# <codecell>
#### UNCOMMENT THIS CELL FOR GOOGLE COLAB EXECUTION
#!pip install geotorch
#import geotorch
#!pip install git+https://github.com/dherrera1911/accuracy_maximization_analysis.git
#from ama_library import *
#!wget -qO ./data/ https://github.com/burgelab/AMA/blob/master/AMAdataDisparity.mat

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
ctgInd = torch.Tensor(ctgInd)
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
# Extract original noise parameters
filterSigmaOri = data.get("var0")
maxRespOri = data.get("rMax")

# <codecell>
##############
#### SET TRAINING PARAMETERS
##############
nFilt = 4 # Number of filters to use
filterSigma = filterSigmaOri / maxRespOri**2 # Variance of filter responses
nEpochs = 50
learningRateBase = 0.004
# Choose loss function
lossFun = nn.CrossEntropyLoss()

# <codecell>
##############
#### FIT AMA WITH DIFFERENT BATCH SIZES
##############
nStim = s.shape[0]
batchFractions = np.array([1/50, 1/20, 1/10, 1/5, 1])
nBatchSizes = batchFractions.size
batchSizeVec = (nStim*batchFractions).astype(int)

filterDict = {"batchSize": [], "filter": [], "loss": [], 'time': [],
        'estimates': [], 'estimateStats': []}

for bs in range(nBatchSizes):
    learningRate = learningRateBase * np.sqrt(10 * batchFractions[bs])
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

# <codecell>
# DEFINE A FUNCTION TO VISUALIZE BINOCULAR FILTERS
def view_filters_bino(f, x=[], title=''):
    plt.title(title)
    nPixels = int(max(f.shape)/2)
    if len(x) == 0:
        x = np.arange(nPixels)
    plt.plot(x, f[:nPixels], label='L', color='red')
    plt.plot(x, f[nPixels:], label='R', color='blue')
    plt.ylim(-0.3, 0.3)

# <codecell>
# Plot the first 2 filters for each batch size
x = np.linspace(start=-30, stop=30, num=nPixels) # x axis in arc min
for bs in range(nBatchSizes):
    plt.subplot(2, nBatchSizes, bs+1)
    # Plot filter 1
    f1 = filterDict["filter"][bs][0,:]
    view_filters_bino(f=f1, x=x, title='size: %i'  %batchSizeVec[bs])
    plt.subplot(2, nBatchSizes, bs+1+nBatchSizes)
    # Plot filter 2
    f2 = filterDict["filter"][bs][1,:]
    view_filters_bino(f=f2, x=x, title=''  %batchSizeVec[bs])
plt.show()

# <codecell>
# Plot the learning curve for each batch size
minLoss = 2.55 # Lower limit of y axis
maxTime = 10   # Upper limit of X axis in the time plot
for bs in range(nBatchSizes):
    plt.subplot(2, nBatchSizes, bs+1)
    loss = filterDict["loss"][bs]
    time = filterDict["time"][bs]
    plt.plot(time, loss)
    plt.ylim(minLoss, 2.98)
    plt.xlim(0, maxTime)
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
    plt.ylim(minLoss, 2.98)
    plt.ylabel('Cross entropy loss')
    plt.xlabel('Epoch')
    if bs>0:
        plt.yticks([])
        plt.ylabel('')
plt.show()

