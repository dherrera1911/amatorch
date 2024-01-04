#####################
#
# This is the most basic test that the AMA model class works. It will
# train the model on a small dataset and then predict on the same dataset.
#
#####################

##############
#### IMPORT PACKAGES
##############
import numpy as np
import matplotlib.pyplot as plt
import torch
import ama_library.ama_class as cl
import ama_library.utilities as au
from torch.distributions.multivariate_normal import MultivariateNormal


##############
# GENERATE AMA DATA
##############
nCtg = 5
nStim = 1000
nDim = 200

# Create some features
nFeatures = 2
freq1 = 3
feat1 = torch.sin(torch.linspace(0, 2*np.pi*freq1, nDim))
freq2 = 2
feat2 = torch.sin(torch.linspace(0, 2*np.pi*freq2, nDim))
freq3 = 1
feat3 = torch.sin(torch.linspace(0, 2*np.pi*freq3, nDim))

# Create covariance matrices for the features in the
# different categories

# Make a rotation matrix
theta = np.pi/(nCtg+2)
rotMat = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])


# Create covariance matrices for the features in the
# different categories
covs = torch.zeros((nCtg, nFeatures, nFeatures))
covs[0,:,:] = torch.diag(torch.tensor([0.2, 0.02]))
for i in range(1, nCtg):
    covs[i,:,:] = rotMat @ covs[i-1,:,:].squeeze() @ rotMat.transpose(0,1)
# Noise parameters
feat3SD = 0.05 # Noisiness of feature 3
noiseSD = 0.01 # Noisiness of white noise

# Create the stimuli
s = torch.zeros((nCtg, nStim, nDim))
ctgInd = torch.zeros((nCtg, nStim))
for i in range(nCtg):
    featureDist = MultivariateNormal(torch.zeros(nFeatures), covs[i,:,:])
    featureSamp = featureDist.sample([nStim])
    noiseSamp = torch.randn(nStim) * feat3SD
    # Add the 3 features together according to their weights
    s[i,:,:] = torch.einsum('n,d->nd', featureSamp[:,0], feat1) + \
               torch.einsum('n,d->nd', featureSamp[:,1], feat2) + \
               torch.einsum('n,d->nd', noiseSamp, feat3)
    s[i,:,:] = s[i,:,:] + torch.randn(nStim, nDim) * noiseSD
    ctgInd[i,:] = i

# Rearrange the stimuli into a single matrix
s = s.reshape((nCtg*nStim, nDim))
ctgInd = ctgInd.reshape((nCtg*nStim))
ctgVal = torch.linspace(0, nCtg-1, nCtg)


##############
# INITIALIZE AMA
##############

ama = cl.AMA_emp(sAll=s, ctgInd=ctgInd, nFilt=2, respNoiseVar=0.05,
                 pixelCov=torch.tensor(0.005), ctgVal=ctgVal,
                 samplesPerStim=5, nChannels=1)

# Change to feature filters
# Put two features into a 2 x nDim matrix
fNew = torch.zeros((2, nDim))
fNew[0,:] = feat1/feat1.norm()
fNew[1,:] = feat2/feat2.norm()

ama.assign_filter_values(fNew=fNew)
ama.update_response_statistics()


##############
# COMPARE INITIALIZATION WITH THE TRUE PARAMETERS
##############

# Apply preprocessing to the stimuli
sPre = ama.preprocess(s=s)
# Apply filters to the stimuli
f = ama.all_filters().detach().clone()
# Apply the filters to the stimuli
responses = torch.einsum('nd,kd->nk', sPre, f)
# Get responses mean and covariance
meanResp = au.category_means(responses, ctgInd)
secondM = au.category_secondM(responses, ctgInd)
covResp = au.secondM_2_cov(secondM, meanResp)

# Compare to class method responses
responses2 = ama.get_responses(s=s, addRespNoise=False)
meanResp2 = au.category_means(responses2.detach().clone(), ctgInd)
secondM2 = au.category_secondM(responses2, ctgInd)
covResp2 = au.secondM_2_cov(secondM2, meanResp2)

diff = ama.respCovNoiseless - covResp

# Plot a stimulus
i = 6
plt.plot(sPre[i,:].squeeze(), label='Preprocessed')
plt.plot(s[i,:].squeeze(), label='Raw')
plt.legend()
plt.show()


##############
# PLOT RESPONSE DISTRIBUTION
##############

responses = ama.get_responses(s=s, addRespNoise=False)
responses = torch.einsum('nd,kd->nk', sPre, fNew)
responses = responses.detach().clone()

ctgPlot = 4
pltInd = ctgInd == ctgPlot
plt.scatter(responses[pltInd,0], responses[pltInd,1])
plt.show()


##############
# INFER STIMULUS CATEGORY
##############

estimates = ama.get_estimates(s=s, method4est='MAP', addRespNoise=True)
correct = estimates == ctgInd
correct = correct.numpy()
print('Accuracy: ' + str(correct.sum()/len(correct)))

