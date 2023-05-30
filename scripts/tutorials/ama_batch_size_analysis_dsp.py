# <markdowncell>

# # Disparity estimation and effect of batch size
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
from torch.utils.data import TensorDataset, DataLoader

# <codecell>
##UNCOMMENT_FOR_COLAB_START##
#!pip install geotorch
#import geotorch
#!pip install git+https://github.com/dherrera1911/accuracy_maximization_analysis.git
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


# <codecell>
##############
#### SET TRAINING PARAMETERS
##############
nFilt = 2   # Number of filters to use
pixelNoiseVar = 0.001  # Variance of pixel noise
respNoiseVar = 0.003  # Variance of filter responses
nEpochsBase = 150
lrGamma = 0.3   # multiplication factor for lr decay
lossFun = au.cross_entropy_loss()
learningRate = 0.01


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
    # Adjust learning parameters to the batch size
    nEpochs = int(nEpochsBase*np.sqrt(batchFractions[bs]))
    lrStepSize = int(nEpochs/3)
    batchSize = int(batchSizeVec[bs])
    # Initialize model with random filters
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
    #opt = torch.optim.SGD(ama.parameters(), lr=0.03)  # SGD
    # fit model
    loss, elapsedTimes = au.fit(nEpochs=nEpochs, model=ama,
            trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
            sAll=s, ctgInd=ctgInd, scheduler=scheduler)
    # Store ama information into dictionary
    filterDict["batchSize"].append(batchSize)
    filterDict["filter"].append(ama.f.detach())
    filterDict["loss"].append(loss)
    filterDict["time"].append(elapsedTimes)
    filterDict["estimates"].append(ama.get_estimates(s))
    filterDict["estimateStats"].append(
            au.get_estimate_statistics(filterDict["estimates"][bs], ctgInd))


# <codecell>
# Plot the first 2 filters for each batch size
x = np.linspace(start=-30, stop=30, num=nPixels) # x axis in arc min
for bs in range(nBatchSizes):
    for nf in range(nFilt):
        plt.subplot(nFilt, nBatchSizes, bs+1+nBatchSizes*nf)
        fPlot = filterDict["filter"][bs][nf,:]
        ap.view_1D_bino_image(inputVec=fPlot, x=x,
                title='size: %i'  %batchSizeVec[bs])
plt.show()


# <codecell>
# Plot the learning curve for each batch size
minLoss = np.min([np.min(arr) for arr in filterDict["loss"]])
maxLoss = np.max([np.max(arr) for arr in filterDict["loss"]])
margin = (maxLoss - minLoss)*0.05

maxTime = 5   # Upper limit of X axis in the time plot
for bs in range(nBatchSizes):
    plt.subplot(2, nBatchSizes, bs+1)
    loss = filterDict["loss"][bs]
    time = filterDict["time"][bs]
    plt.plot(time, loss)
    plt.ylim(minLoss-margin, maxLoss+margin)
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
    plt.ylim(minLoss-margin, maxLoss+margin)
    plt.ylabel('Cross entropy loss')
    plt.xlabel('Epoch')
    if bs>0:
        plt.yticks([])
        plt.ylabel('')
plt.show()

