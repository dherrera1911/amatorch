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
from torch.utils.data import TensorDataset, DataLoader


# <codecell>
# Download data and install packages
##UNCOMMENT_FOR_COLAB_START##
#!pip install geotorch
#import geotorch
#!pip install pymanopt
#import pymanopt as pm
#!pip install git+https://github.com/dherrera1911/accuracy_maximization_analysis.git
#from ama_library import *
#!mkdir data
#!wget -O ./data/AMAdataDisparity.mat https://github.com/burgelab/AMA/blob/master/AMAdataDisparity.mat?raw=true
##UNCOMMENT_FOR_COLAB_END##


# <codecell>
##### COMMENT THIS CELL WHEN USING GOOGLE COLAB
import ama_library.ama_class as cl
import ama_library.utilities as au
import ama_library.plotting as ap
import pymanopt as pm
import geomstats as gs
from geomstats.geometry.spd_matrices import SPDMatrices


# <codecell>
##############
#### LOAD AMA DATA
##############
# Load ama struct from .mat file into Python
data = spio.loadmat('./data/ama_dsp_noiseless.mat')
# Extract stimuli and category vectors
s, ctgInd, ctgVal = au.unpack_matlab_data(matlabData=data)
nPixels = int(s.shape[1]/2)

# <codecell>
##########################
#
# OBTAIN COVARIANCE MATRICES
#
##########################

##############
#### SET TRAINING PARAMETERS
##############
nPairs = 2   # Number of filters to use
pixelNoiseVar = 0.001  # Input pixel noise variance
respNoiseVar = 0.003  # Filter response noise variance
nEpochs = 30
lrGamma = 0.5   # multiplication factor for lr decay
lossFun = au.cross_entropy_loss()
#lossFun = au.kl_loss()
learningRate = 0.05
lrStepSize = 10
batchSize = 256
nSeeds = 2
nFilt = nPairs * 2


# <codecell>
##############
#### TRAINED MODEL COVARIANCES
##############

# Define model
ama = cl.Isotropic(sAll=s, ctgInd=ctgInd, nFilt=2,
        respNoiseVar=respNoiseVar, pixelVar=pixelNoiseVar, ctgVal=ctgVal, 
        filtNorm='broadband', respCovPooling='pre-filter')

# Put training data into Torch data loader tools
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
# FIT MODEL TO DATA
loss, elapsedTimes = au.fit_by_pairs(nEpochs=nEpochs, model=ama,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt_fun=opt_fun,
        nPairs=nPairs, sAll=s, ctgInd=ctgInd, scheduler_fun=scheduler_fun,
        seedsByPair=nSeeds)

# EXTRACT RESPONSE COVARIANCES FOR TRAINED FILTERS
respCov = ama.respCov.detach().numpy()
fLearned = ama.f.detach().clone().numpy()


# <codecell>
##############
#### UNTRAINED MODEL COVARIANCES
##############
amaRand = cl.Isotropic(sAll=s, ctgInd=ctgInd, nFilt=nFilt,
        respNoiseVar=respNoiseVar, pixelVar=pixelNoiseVar, ctgVal=ctgVal, 
        filtNorm='broadband', respCovPooling='pre-filter')
# Extract response covariances for trained filters
respRandomCovs = amaRand.respCov.detach().numpy()
fRandom = amaRand.f.detach().clone().numpy()


# <codecell>
##############
#### PCA FILTER COVARIANCES
##############

# Get the overall dataset covariance to get the PCA filters
stimCovMean = torch.mean(ama.stimCov, dim=0)
# Get eigenvectors of overall dataset covariance. These are PCA filters
eigVals, eigVecs = torch.linalg.eig(stimCovMean)
fPCA = eigVecs.real[:,0:nFilt].transpose(0,1)
# Use the PCA filters to get the class-specific response covariance
pcaCovs = torch.einsum('fd,jdb,gb->jfg', fPCA, ama.stimCov, fPCA)
pcaCovs = pcaCovs + ama.respNoiseCov.unsqueeze(0).repeat(ama.nClasses, 1, 1)
pcaCovs = pcaCovs.numpy()

# <codecell>
# Put the learned filters into a list for tidier plotting
fList = [fLearned, fRandom, fPCA]
namesList = ['Learned filters', 'Random filters', 'PCA filters', 'Stimuli']

# <codecell>
##########################
#
# MANIFOLD ANALYSIS
#
##########################

# Define function to compute angles formed by each matrix with its 2 neighbors
def compute_average_dist(inputMat, inputMan):
    distVec = np.zeros(inputMat.shape[0]-2)
    for c in range(inputMat.shape[0]-2):
        pointCenter = inputMat[c+1,:,:]
        pointPrev = inputMat[c,:,:]
        pointNext = inputMat[c+2,:,:]
        dist1 = inputMan.dist(pointCenter, pointPrev)
        dist2 = inputMan.dist(pointCenter, pointNext)
        distVec[c] = np.mean([dist1, dist2])
    return distVec

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
# INITIALIZE MANIFOLD STRUCTURE TO ANALYZE GEOMETRY
# SPDM
manifold = pm.manifolds.positive_definite.SymmetricPositiveDefinite(ama.nFilt, k=1)
manifoldStim = pm.manifolds.positive_definite.SymmetricPositiveDefinite(ama.nDim, k=1)
# PSDM
#manifold = pm.manifolds.psd.PSDFixedRank(ama.nFilt, k=ama.nFilt)
#manifoldStim = pm.manifolds.psd.PSDFixedRank(ama.nDim, k=ama.nDim)
# Euclidean
#manifold = pm.manifolds.euclidean.Euclidean((nFilt, nFilt))
#manifoldStim = pm.manifolds.euclidean.Euclidean((ama.nDim, ama.nDim))


# <codecell>
### Compute average distance between a class and its neighbors
respCovDist = compute_average_dist(respCov, manifold)
respRandomCovDist = compute_average_dist(respRandomCovs, manifold)
respPCACovDist = compute_average_dist(pcaCovs, manifold)
stimCovDist = compute_average_dist(np.array(ama.stimCov), manifoldStim)
# Put distances into list for tidier plotting
distancesList = [respCovDist, respRandomCovDist, respPCACovDist, stimCovDist]
ymaxDist = np.max(distancesList)

# <codecell>
### Compute angles between classes
# Ama filter responses
respCovAng = compute_angles(respCov, manifold)
# Random filter responses
respRandomCovAng = compute_angles(respRandomCovs, manifold)
# PCA filter responses
respPCACovAng = compute_angles(pcaCovs, manifold)
# Raw stimuli covariances
stimCovAng = compute_angles(np.array(ama.stimCov), manifoldStim)
# Put angles into list for tidier plotting
anglesList = [respCovAng, respRandomCovAng, respPCACovAng, stimCovAng]

# <codecell>
### PLOT FILTERS
nModels = len(namesList)
x = np.linspace(start=-30, stop=30, num=nPixels) # x axis in arc min
for n in range(nModels-1):
    for nf in range(nFilt):
        fPlot = fList[n]
        plt.subplot(nFilt, nModels, n+1+nModels*nf)
        titleStr = namesList[n]
        if (nf>0):
            titleStr = ''
        ap.view_1D_bino_image(inputVec=fPlot[nf,:], x=x, title=titleStr)
        plt.yticks([])
        plt.xlabel('Visual field (arcmin)')

plt.show()

# <codecell>
### PLOT RESULTS 
disparities = ctgVal[1:-1]
# Plot distances
for n in range(nModels):
    plt.subplot(2, nModels, n+1)
    plt.title(namesList[n]) 
    if (n==0):
        plt.ylabel('Distance between classes')
    else:
        plt.yticks([])
    plt.plot(disparities, distancesList[n])
    plt.ylim([0, ymaxDist*1.1])
    plt.subplot(2, nModels, n+(nModels+1))
    if (n==0):
        plt.ylabel('Angle between classes')
    else:
        plt.yticks([])
    plt.plot(disparities, anglesList[n])
    plt.ylim(0, 90)
    plt.xlabel('Disparity (arcmin)')

plt.show()


# <codecell>
from sklearn.manifold import Isomap

# Define function to compute matrix of pairwise distances
def compute_dist_matrix(inputMat, inputMan):
    nClasses = inputMat.shape[0]
    distMat = np.zeros((nClasses, nClasses))
    for c in range(nClasses):
        point1 = inputMat[c,:,:]
        for d in range(nClasses):
            point2 = inputMat[d,:,:]
            distMat[c,d] = inputMan.dist(point1, point2)
    # make matrix symmetric, taking largest distance in unmatched points
    distMat = np.maximum(distMat, distMat.transpose())
    return distMat

# Get pairwise distance matrix for trained filter responses
distMatResp = compute_dist_matrix(respCov, manifold)
distMatRand = compute_dist_matrix(respRandomCovs, manifold)
distMatPCA = compute_dist_matrix(pcaCovs, manifold)
distMatStim = compute_dist_matrix(np.array(ama.stimCov), manifoldStim)
distMatList = [distMatResp, distMatRand, distMatPCA,
        distMatStim] # put into list for nicer code

# Plot distance matrices
for n in range(nModels):
    plt.subplot(1,nModels,n+1)
    plt.title(namesList[n]) 
    plt.imshow(distMatList[n], zorder=2, cmap='Blues', interpolation='nearest')
    plt.colorbar();
plt.show()

from mpl_toolkits import mplot3d

isomap_model = Isomap(n_neighbors=6, n_components=3, metric='precomputed')
out = [None] * nModels

for n in range(nModels):
    out[n] = isomap_model.fit_transform(distMatList[n])
    ax = plt.subplot(1,nModels,n+1, projection='3d')
    plt.title(namesList[n]) 
    ax.scatter3D(out[n][:,0], out[n][:,1], out[n][:,2], c=ama.ctgVal)
    ax.plot3D(out[n][:,0], out[n][:,1], out[n][:,2])
    plt.axis('equal');
plt.show()


## Look into geometries of geomstats. Information geometry
#from geomstats.information_geometry.normal import NormalDistributions
#import geomstats





