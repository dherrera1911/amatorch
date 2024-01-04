#####################
#
# This is the most basic test that the AMA model class works. It will
# train the model on a small dataset and then predict on the same dataset.
#
#####################

##############
#### IMPORT PACKAGES
##############
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import ama_library.ama_class as cl
import ama_library.utilities as au


##############
# LOAD AMA DATA
##############

# Load data from csv files
s = torch.tensor(np.loadtxt('./test_data/dspStim.csv', delimiter=','))
s = s.transpose(0,1)
# Change s dtype to Double
s = s.float()
ctgInd = np.loadtxt('./test_data/dspCtg.csv', delimiter=',')
# Change ctgInd to integer tensor and make 0-indexed
ctgInd = torch.tensor(ctgInd, dtype=torch.int64) - 1
ctgVal = torch.tensor(np.loadtxt('./test_data/dspVal.csv', delimiter=','))

##############
# Initialize AMA model
##############

samplesPerStim = 5
respNoiseVar = torch.tensor(0.003)
pixelCov = torch.tensor(0.005)

#ama = cl.AMA_emp(sAll=s, ctgInd=ctgInd, nFilt=2, respNoiseVar=respNoiseVar,
#                 pixelCov=pixelCov, ctgVal=ctgVal,
#                 samplesPerStim=samplesPerStim, nChannels=2)

ama = cl.AMA_qmiso(sAll=s, ctgInd=ctgInd, nFilt=2, respNoiseVar=respNoiseVar,
                 pixelVar=pixelCov, ctgVal=ctgVal)


##############
# Train AMA model
##############

# Models parameters
nEpochs = 50
lrGamma = 0.7   # multiplication factor for lr decay
#lossFun = au.kl_loss()
lossFun = au.cross_entropy_loss()
learningRate = 0.05
lrStepSize = 5
batchSize = 256

# Put data into Torch data loader tools
trainDataset = TensorDataset(s, ctgInd)
# Batch loading and other utilities 
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
# Optimizer and scheduler
opt = torch.optim.Adam(ama.parameters(), lr=learningRate)
sch = torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize, gamma=lrGamma)

# Fit model
loss, tstLoss, elapsedTimes = au.fit(nEpochs=nEpochs, model=ama,
                            trainDataLoader=trainDataLoader, lossFun=lossFun,
                            opt=opt, scheduler=sch, sTst=s, ctgIndTst=ctgInd)

##############
# Plot results
##############

# Plot loss
plt.plot(loss)
plt.show()

# Plot filters
f = ama.f.detach().clone()
pix = int(f.shape[1]/2)
for sp in range(2):
    plt.subplot(1,2,sp+1)
    plt.plot(f[sp,:pix], label='L')
    plt.plot(f[sp,pix:], label='R')
plt.show()



