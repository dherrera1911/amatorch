import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# Functions for plotting the filters
def unvectorize_filter(fIn, frames=15, pixels=30):
    nFilt = fIn.shape[0]
    matFilt = fIn.reshape(nFilt, 2, frames, pixels)
    matFilt = matFilt.transpose(1,2).reshape(nFilt, frames, pixels*2)
    return matFilt

# Plot filters
def view_filters_bino_video(fIn, frames=15, pixels=30):
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
    estimatesMeans = torch.zeros(ctgInd.max()+1)
    estimatesSD = torch.zeros(ctgInd.max()+1)
    lowCI = torch.zeros(ctgInd.max()+1)
    highCI = torch.zeros(ctgInd.max()+1)
    quantiles = torch.tensor(quantiles, dtype=torch.float64)
    for cl in ctgInd.unique():
        levelInd = [i for i, j in enumerate(ctgInd) if j == cl]
        estimatesMeans[cl] = estimates[levelInd].mean()
        estimatesSD[cl] = estimates[levelInd].std()
        (lowCI[cl], highCI[cl]) = torch.quantile(estimates[levelInd], quantiles)
    return {'estimateMean': estimates, 'estimateSD': estimatesSD,
            'lowCI': lowCI, 'highCI': highCI}

# Define loop function to train the model
def fit(nEpochs, model, trainDataLoader, lossFun, opt):
    trainingLoss = np.zeros(nEpochs+1)
    elapsedTime = np.zeros(nEpochs+1)
    # Get the loss of the full dataset stored in the data loader
    trainingLoss[0] = lossFun(model.get_posteriors(trainDataLoader.dataset.tensors[0]),
            trainDataLoader.dataset.tensors[1]).detach()
    print('Initial loss: ', trainingLoss[0])
    opt.zero_grad()
    start = time.time()
    for epoch in range(nEpochs):
        for sb, ctgb in trainDataLoader:
            # Generate predictions for batch sb, returned by trainDataLoader 
            model.update_response_statistics()
            pred = model.get_posteriors(sb)
            loss = lossFun(pred, ctgb)  # Compute loss
            loss.backward()             # Compute gradient
            opt.step()                  # Take one step
            opt.zero_grad()             # Restart gradient
        # Print model loss
        trainingLoss[epoch+1] = lossFun(model.get_posteriors(trainDataLoader.dataset.tensors[0]),
                trainDataLoader.dataset.tensors[1]).detach()
        trainingDiff = trainingLoss[epoch+1] - trainingLoss[epoch]
        print('Epoch: %d |  Training loss: %.4f  |  Loss change: %.4f' %
                (epoch+1, trainingLoss[epoch+1], trainingDiff))
        end = time.time()
        elapsedTime[epoch+1] = end - start
    # Do the final response statistics update
    model.update_response_statistics()
    return trainingLoss, elapsedTime


