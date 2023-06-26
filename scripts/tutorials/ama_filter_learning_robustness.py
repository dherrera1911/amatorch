# <markdowncell>

# #Disparity estimation and reproducibility of learned filters
# 
# To understand the properties of the ideal observer models for natural images,
# it is important to first know how reproducible the learned models are.
# Here we train the AMA model several times with different seeds and compare
# the filters learned across runs. The filters are learned in pairs, which
# seems to improve the reproducibility of the model.
# 
# We also see the results when using a training procedure that aims to generate
# more reproducible results. This procedure trains each pair of filters several
# times using different seeds, and then selects the best of the learned filters
# at each step. This procedure allows for high reproducibility in the filters
# learned for the disparity estimation task, as will be shown.

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
import ama_library.plotting as ap


# <codecell>
##############
#### LOAD AMA DATA
##############
# Load ama struct from .mat file into Python
data = spio.loadmat('./data/ama_dsp_noiseless.mat')
# Extract stimuli and category vectors
s, ctgInd, ctgVal = au.unpack_matlab_data(matlabData=data)
nPixels = int(s.shape[1]/2)


# <markdowncell>
# ## SEE BASELINE FILTER VARIABILITY
# 
# Train the same model several times using different seeds,
# and compare the filters learned.

# <codecell>
##############
#### Set the parameters for training the models
##############

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
loss = [None] * nModels  # List to save the training loss of each model
finalLosses = np.zeros((nModels, nPairs))  # Array to save the final loss of each model
elapsedTimes = [None] * nModels  # List with the training times of each model
filters = [None] * nModels  # List with the filters learned for each model
# Loop over the number of models to train
for n in range(nModels):
    ama = cl.Isotropic(sAll=s, ctgInd=ctgInd, nFilt=2,
            respNoiseVar=respNoiseVar, pixelVar=pixelNoiseVar, ctgVal=ctgVal, 
            filtNorm='broadband', respCovPooling='pre-filter')
    loss[n], elapsedTimes[n] = au.fit_by_pairs(nEpochs=nEpochs, model=ama,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt_fun=opt_fun,
        nPairs=nPairs, sAll=s, ctgInd=ctgInd, scheduler_fun=scheduler_fun)
    filters[n] = ama.fixed_and_trainable_filters().detach().clone()
    for p in range(nPairs):
        finalLosses[n, p] = loss[n][p][-1]

# <markdowncell>
# 
# Print the final loss of the model for each pair of learned filters.
# Each column shows loss for a different filter pair. Each row shows
# the loss for a different model.
#
# We see that there is some variability in the final loss of the models

# <codecell>
##############
#### Print the resulting losses
##############
# Print the loss of the model after each pair of filters is learned.
# Columns indicate the pair of filters, and rows indicate the model instance
print(finalLosses)

# <markdowncell>
# 
# Show all the filters learned for each model.
# Each column shows loss for a different filter pair. Each row shows
# the loss for a different model.
#
# We see that there is some variability in the final loss of the models

# <codecell>
# Plot the learned filters
nFilt = nPairs * 2
fig, axs = plt.subplots(nModels, nFilt)
for n in range(nModels):
    for nf in range(nFilt):
        ax = axs[n, nf]
        plt.subplot(nModels, nFilt, n*nFilt + nf + 1)
        ap.view_1D_bino_image(filters[n][nf,:])
        ax.set_yticks([])
        ax.set_xticks([])
        # Add column label at the top of each column
        if n == 0:
            ax.set_title(f"Filter {nf}")
        # Add row label to the left of each row
        if nf == 0:
            ax.set_ylabel(f"Model {n}", rotation=90, ha='center', va='center')
fig.tight_layout(rect=(0,0,1,0.95))
fig.suptitle("Same model parameters, different seeds", fontsize=14, y=0.98)
plt.show()


# <markdowncell>
# ## TRAINING REGIME FOR STABLE MODELS
#
# Because it is desirable for our analyses of ideal observers
# to have reproducible models that consistently converge to the
# same "optimal solution", we implement a training regime to
# help finding the best model for the task.
#
# This training regime involves training each pair of filters
# several times from different seeds, and choosing the pair of
# filters that minimize the loss. This procedure should reduce the
# effects of training randomness on the resulting model. We
# test the reproducibility of the models trained with this procedure.
#
# We will see that with this procedure, the models learned for
# the disparity estimation task are highly consistent


# <codecell>
##############
#### See filter variability when we train several
#### filters at each step and choose the best performing one
##############
nSeeds = 5  # Number of seeds that are trained for each pair of filters in a model train
nModels = 3  # Number of models to train with the procedure, to see stability
loss = [None] * nModels
finalLosses = np.zeros((nModels, nPairs))
elapsedTimes = [None] * nModels
filters = [None] * nModels

for n in range(nModels):
    ama = cl.Isotropic(sAll=s, ctgInd=ctgInd, nFilt=2,
            respNoiseVar=respNoiseVar, pixelVar=pixelNoiseVar, ctgVal=ctgVal, 
            filtNorm='broadband', respCovPooling='pre-filter')
    loss[n], elapsedTimes[n] = au.fit_by_pairs(nEpochs=nEpochs, model=ama,
        trainDataLoader=trainDataLoader, lossFun=lossFun, opt_fun=opt_fun,
        nPairs=nPairs, sAll=s, ctgInd=ctgInd, scheduler_fun=scheduler_fun,
        seedsByPair=nSeeds)
    filters[n] = ama.fixed_and_trainable_filters().detach().clone()
    for p in range(nPairs):
        finalLosses[n, p] = loss[n][p][-1]

# Print the loss of the model after each pair of filters is learned.
# Columns indicate the pair of filters, and rows indicate the model instance
print(finalLosses)


# <codecell>
# Plot filters learned by selecting the best filters at each pair
nFilt = nPairs * 2
fig, axs = plt.subplots(nModels, nFilt)
for n in range(nModels):
    for nf in range(nFilt):
        ax = axs[n, nf]
        plt.subplot(nModels, nFilt, n*nFilt + nf + 1)
        ap.view_1D_bino_image(filters[n][nf,:])
        ax.set_yticks([])
        ax.set_xticks([])
        # Add column label at the top of each column
        if n == 0:
            ax.set_title(f"Filter {nf}")
        # Add row label to the left of each row
        if nf == 0:
            ax.set_ylabel(f"Model {n}", rotation=90, ha='center', va='center')
fig.tight_layout(rect=(0,0,1,0.95))
fig.suptitle("Filters selected from best seed for each model", fontsize=14, y=0.98)
plt.show()


# <markdowncell>
# ## TESTING GROUND FOR TESTING EFFECT OF PARAMETERS ON STABILITY
#
# Some model parameters may lead to higher reproducibility, or
# to good reproducibility at faster speeds. Below several models
# are trained with different parameters.


# <codecell>
##############
#### Try out different optimization parameters to see if there's
#### differences in the resulting filter variability
##############

nModels = 2
nPairs = 3   # Numbers of pairs of filters to learn
nEpochs = 10
lrGamma = 0.5   # multiplication factor for lr decay
learningRate = 0.02
lrStepSize = 10

batchSize = [256, 1024]
learningRate = [0.04, 0.01]
lrGamma = [0.8, 0.5]

learnDict = {'batchSize': [], 'learningRate': [], 'lrGamma': [],
        'rep': [], 'filters': [], 'finalLosses': []}
for bs in range(len(batchSize)):
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize[bs], shuffle=True)
    for lr in range(len(learningRate)):
    #    lrDict = bsDict.copy()
        def opt_fun(model):
            return torch.optim.Adam(model.parameters(), lr=learningRate[lr])
        for g in range(len(lrGamma)):
    #        gDict = lrDict.copy()
            def scheduler_fun(opt):
                return torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize, gamma=lrGamma[g])
            for n in range(nModels):
    #            nDict = gDict.copy()
                ama = cl.Isotropic(sAll=s, ctgInd=ctgInd, nFilt=2,
                        respNoiseVar=respNoiseVar, pixelVar=pixelNoiseVar, ctgVal=ctgVal,
                        filtNorm='broadband', respCovPooling='pre-filter')
                loss, elapsedTimes = au.fit_by_pairs(nEpochs=nEpochs, model=ama,
                    trainDataLoader=trainDataLoader, lossFun=lossFun,
                    opt_fun=opt_fun, nPairs=nPairs, sAll=s, ctgInd=ctgInd,
                    scheduler_fun=scheduler_fun)
                filters = ama.fixed_and_trainable_filters().detach().clone()
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
nConditions = len(learnDict['batchSize'])
plt.scatter(learnDict['batchSize'] + np.random.randint(-sc, sc, nConditions),
        learnDict['finalLosses'][:,1],
        c=learnDict['learningRate'],
        s=learnDict['lrGamma']**2*100)
plt.xlabel('Batch size')
# Add colorbar with title
cbar = plt.colorbar()
cbar.set_label('LR')
# Create a legend for the size of the dots
size_legend_title = 'LR adaptation'
sizes =  np.array(lrGamma)*100 # Adjust these values based on your actual data
size_labels = [str(size/100) for size in sizes]
# Create legend handles
legend_handles = [plt.scatter([], [], c='gray', s=size, label=size_label)
        for size, size_label in zip(sizes, size_labels)]
# Add legend for size
plt.legend(handles=legend_handles, title=size_legend_title, loc='upper right')
plt.show()


# <codecell>
# Plot filters learned for one combination of parameters
nFilt = nPairs * 2
inds = np.logical_and.reduce((learnDict['batchSize']==1024,
        learnDict['learningRate']==0.04,
        learnDict['lrGamma']==0.8))
filters = learnDict['filters'][inds,:,:]
fig, axs = plt.subplots(nModels, nFilt)
for n in range(nModels):
    for nf in range(nFilt):
        ax = axs[n, nf]
        plt.subplot(nModels, nFilt, n*nFilt + nf + 1)
        ap.view_1D_bino_image(filters[n][nf,:])
        ax.set_yticks([])
        ax.set_xticks([])
        # Add column label at the top of each column
        if n == 0:
            ax.set_title(f"Filter {nf}")
        # Add row label to the left of each row
        if nf == 0:
            ax.set_ylabel(f"Model {n}", rotation=90, ha='center', va='center')
fig.tight_layout(rect=(0,0,1,0.95))
fig.suptitle("Filters selected from best seed for each model", fontsize=14, y=0.98)
plt.show()

