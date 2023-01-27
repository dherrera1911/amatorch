# <markdowncell>

# # Geometrical analysis of disparity estimation statistics
# 
# Train AMA on the task of disparity estimation. Analyze the
# distances and angles between the covariance matrices of
# each class, in the manifold of symmetric positive definite matrices

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
import pymanopt as pm

# <codecell>
#### UNCOMMENT THIS CELL FOR GOOGLE COLAB EXECUTION
#!pip install geotorch
#import geotorch
#!pip install pymanopt
#import pymanopt as pm
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
# Extract original noise parameters
filterSigmaOri = data.get("var0").flatten()
maxRespOri = data.get("rMax").flatten()

# <codecell>
##############
#### SET TRAINING PARAMETERS
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
####  TRAIN THE MODEL
##############

# Define model
amaPy = AMA(sAll=s, nFilt=nFilt, ctgInd=ctgInd, filterSigma=filterSigma,
        ctgVal=ctgVal)

# Extract the initial random response covariances
randomRespCovs = amaPy.respCovs.detach().numpy()

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

# <codecell>
# fit model to data
loss, elapsedTimes = fit(nEpochs=nEpochs, model=amaPy,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
        scheduler=scheduler)
plt.plot(elapsedTimes, loss)
plt.show()

# Extract response covariances for trained filters
respCovs = amaPy.respCovs.detach().numpy()

# <codecell>
##############
#### GET COVARIANCE MATRICES OF PCA FILTER RESPONSES
##############
# Reduce dimansionality of stimulus covs so they are not singular
pcaDim = nFilt
stimCovs = amaPy.stimCovs.detach()
u, a, pcaFilt = np.linalg.svd(s)
pcaFilt = torch.from_numpy(pcaFilt[0:pcaDim,:])
pcaCovs = torch.einsum('fd,jdb,gb->jfg', pcaFilt, stimCovs, pcaFilt)
pcaCovs = pcaCovs.numpy()

# <codecell>
###########
### MANIFOLD ANALYSIS
###########
# Initialize manifold structures to analyze geometry
# SPDM
manifold = pm.manifolds.positive_definite.SymmetricPositiveDefinite(amaPy.nFilt, k=1)
# PSDM
#manifold = pm.manifolds.psd.PSDFixedRank(amaPy.nFilt, k=2)

# <codecell>
### Compute distances between classes
respCovDist = np.zeros(amaPy.nClasses-1)
respRandomCovDist = np.zeros(amaPy.nClasses-1)
stimCovDist = np.zeros(amaPy.nClasses-1)
for c in range(amaPy.nClasses-1):
    respCovDist[c] = manifold.dist(respCovs[c,:,:], respCovs[c+1,:,:])
    respRandomCovDist[c] = manifold.dist(randomRespCovs[c,:,:], randomRespCovs[c+1,:,:])
    stimCovDist[c] = manifold.dist(pcaCovs[c,:,:], pcaCovs[c+1,:,:])

ymaxDist = np.max([respRandomCovDist, respCovDist, stimCovDist])

# <codecell>
# Define function to compute angles formed by each matrix with its 2 neighbors
def compute_angles(inputMat, inputMan):
    angleVec = np.zeros(inputMat.shape[0]-2)
    for c in range(inputMat.shape[0]-2):
        pointCenter = inputMat[c+1,:,:]
        pointPrev = inputMat[c,:,:]
        pointNext = inputMat[c+2,:,:]
        # Get the vectors between a point and its two adjacent points
        tangentVec1 = inputMan.to_tangent_space(pointCenter, pointPrev - pointCenter)
        normVec1 = inputMan.norm(pointCenter, tangentVec1)
        tangentVec2 = inputMan.to_tangent_space(pointCenter, pointCenter - pointNext)
        normVec2 = inputMan.norm(pointCenter, tangentVec2)
        # Compute angle between vectors (inner product divided by norm product)
        cosAngle = np.divide(inputMan.inner_product(pointCenter, tangentVec1, tangentVec2),
            (normVec1 * normVec2))
        angle = np.arccos(cosAngle) * 360 / (2*np.pi)
        if angle > 90:
            angle = 180 - angle
        angleVec[c] = angle
    return angleVec

# <codecell>
### Compute angles between classes
for c in range(amaPy.nClasses-2):
    # Ama filter responses
    respCovAng = compute_angles(respCovs, manifold)
    # Random filter responses
    respRandomCovAng = compute_angles(randomRespCovs, manifold)
    # PCA filter responses
    stimCovAng = compute_angles(pcaCovs, manifold)

# <codecell>
### PLOT RESULTS 
# Plot distances
plt.subplot(2,3,1)
plt.title('Trained filters')
plt.ylabel('Distance between classes')
plt.plot(respCovDist)
plt.ylim([0, ymaxDist*1.1])
plt.subplot(2,3,2)
plt.title('Random filters')
plt.plot(respRandomCovDist)
plt.ylim([0, ymaxDist*1.1])
plt.yticks([])
plt.subplot(2,3,3)
plt.title('PCA filters')
plt.plot(stimCovDist)
plt.ylim([0, ymaxDist*1.1])
plt.yticks([])
# Plot angles
plt.subplot(2,3,4)
plt.ylabel('Angle between classes')
plt.plot(respCovAng)
plt.ylim(0, 90)
plt.subplot(2,3,5)
plt.plot(respRandomCovAng)
plt.ylim(0, 90)
plt.yticks([])
plt.subplot(2,3,6)
plt.plot(stimCovAng)
plt.ylim(0, 90)
plt.yticks([])
plt.show()

