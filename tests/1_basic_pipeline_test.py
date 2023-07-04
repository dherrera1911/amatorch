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
import ama_library.plotting as ap


##############
# LOAD AMA DATA
##############
# Load ama struct from .mat file into Python
data = spio.loadmat('./data/ama_dsp_noiseless.mat')
s, ctgInd, ctgVal = au.unpack_matlab_data(matlabData=data)


##############
# SET THE PARAMETERS FOR TRAINING THE MODEL
##############
# Models parameters
nPairs = 3   # Number of filters to use
pixelNoiseVar = 0.001  # Input pixel noise variance
respNoiseVar = 0.003  # Filter response noise variance
nEpochs = 20
lrGamma = 0.5   # multiplication factor for lr decay
#lossFun = au.cross_entropy_loss()
lossFun = au.kl_loss()
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


##############
# Train Empirical model
##############

samplesPerStim = 5
ama = cl.Empirical(sAll=s, ctgInd=ctgInd, nFilt=2, respNoiseVar=respNoiseVar,
        pixelCov=torch.tensor(pixelNoiseVar), ctgVal=ctgVal, filtNorm='broadband',
        respCovPooling='pre-filter', samplesPerStim=samplesPerStim, nChannels=2)

loss, elapsedTimes = au.fit_by_pairs(nEpochs=nEpochs, model=ama,
    trainDataLoader=trainDataLoader, lossFun=lossFun, opt_fun=opt_fun,
    nPairs=nPairs, sAll=s, ctgInd=ctgInd, scheduler_fun=scheduler_fun)


##############
# Plot the results
##############

ap.view_all_filters_1D_bino_image(ama)
plt.show()

# Get estimates without interpolation
resp = ama.get_responses(s, addRespNoise=True)
likelihoods = ama.resp_2_log_likelihood(resp)
posteriors = ama.log_likelihood_2_posterior(likelihoods).detach()
estimates = ama.posterior_2_estimate(posteriors, method4est='MAP')

# Convert estimates into response statistics
statistics = au.get_estimate_statistics(estimates, ctgInd, quantiles=[0.16, 0.84])
errorInterval = torch.cat((statistics['lowCI'].unsqueeze(0), 
                            statistics['highCI'].unsqueeze(0)), dim=0)

# Plot the results
ap.plot_estimate_statistics_sd(estMeans=statistics['estimateMean'],
                               errorInterval=errorInterval)

ctg2plot = au.subsample_categories(nCtg=len(np.unique(ctgInd)),
                                   subsampleFactor=4)

ap.plot_posteriors(posteriors=posteriors, ctgInd=ctgInd, ctg2plot=ctg2plot,
                ctgVal=ctgVal, traces2plot=100, quantiles=[0.16, 0.84],
                showPlot=True)


# Get estimates with interpolation
nPoints = 32
ama.interpolate_class_statistics(nPoints=nPoints, method='geodesic',
                                 metric='Euclidean')
resp2 = ama.get_responses(s, addRespNoise=True)
likelihoods2 = ama.resp_2_log_likelihood(resp2)
posteriors2 = ama.log_likelihood_2_posterior(likelihoods2).detach()
estimates2 = ama.posterior_2_estimate(posteriors2, method4est='MAP')

# Convert estimates into response statistics
statistics2 = au.get_estimate_statistics(estimates2, ctgInd,
                                         quantiles=[0.16, 0.84])
errorInterval2 = torch.cat((statistics2['lowCI'].unsqueeze(0), 
                            statistics2['highCI'].unsqueeze(0)), dim=0)

# Plot the results
ap.plot_estimate_statistics_sd(estMeans=statistics2['estimateMean'],
                               errorInterval=errorInterval2)

ctgValInterp = au.interpolate_category_values(ctgVal, nPoints=nPoints)
ctgIndInterp = ctgInd*(nPoints+1)
ctg2plotInterp = ctg2plot*(nPoints+1)

ctg2plot = au.subsample_categories(nCtg=len(np.unique(ctgInd)),
                                   subsampleFactor=4)

ap.plot_posteriors(posteriors=posteriors2, ctgInd=ctgIndInterp,
                   ctg2plot=ctg2plotInterp, ctgVal=ctgValInterp,
                   traces2plot=100, quantiles=[0.16, 0.84], showPlot=True)


