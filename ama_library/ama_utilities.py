import numpy as np
import torch
import matplotlib.pyplot as plt

# Functions for plotting the filters
def unvectorize_filter(fIn, frames=15, pixels=30):
    nFilt = fIn.shape[0]
    matFilt = fIn.reshape(nFilt, 2, frames, pixels)
    matFilt = matFilt.transpose(1,2).reshape(nFilt, frames, pixels*2)
    return matFilt

# Plot filters
def view_filters(fIn, frames=15, pixels=30):
    matFilt = unvectorize_filter(fIn, frames=frames, pixels=pixels)
    nFilters = matFilt.shape[0]
    for k in range(nFilters):
        plt.subplot(nFilters, 1, k+1)
        plt.imshow(matFilt[k,:,:].squeeze())
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

# Function that turns posteriors into estimate averages, SDs and CIs
def get_estimate_statistics(estimates, ctgInd, quantiles=[0.05, 0.95]):
    # Compute means and stds for each true level of the latent variable
    estimates = torch.zeros(ctgInd.max()+1)
    estimatesSD = torch.zeros(ctgInd.max()+1)
    lowCI = torch.zeros(ctgInd.max()+1)
    highCI = torch.zeros(ctgInd.max()+1)
    quantiles = torch.Tensor(quantiles)
    for cl in ctgInd.unique():
        levelInd = [i for i, j in enumerate(ctgInd) if j == cl]
        estimates[cl] = estimateMat[levelInd].mean()
        estimatesSD[cl] = estimateMat[levelInd].std()
        (lowCI[cl], highCI[cl]) = torch.quantile(estimateMat[levelInd], quantiles)
    return (estimates, estimatesSD, lowCI, highCI)


