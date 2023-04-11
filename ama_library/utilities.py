import numpy as np
import torch
from torch import optim
from torch import fft as fft
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import patches, colors, cm
from ama_library import quadratic_moments as qm
import time

##################################
##################################
#
## FUNCTIONS FOR FITTING AMA MODELS
#
##################################
##################################
#
# This group of functions take an ama model, and some inputs
# such as the loss function, and do the training loop.
# Different types of training are available, such as training
# the filters in pairs, or with multiple seeds


# Define loop function to train the model
def fit(nEpochs, model, trainDataLoader, lossFun, opt, sAll,
        ctgInd, scheduler=None, addStimNoise=True,
        addRespNoise=True):
    """ Fit AMA model using the posterior distribuions generated by the model.
    #
    Arguments:
    - Epochs: Number of epochs. Integer.
    - model: AMA model object.
    - trainDataLoader: Data loader generated with torch.utils.data.DataLoader.
    - lossFun: Loss function that uses posterior distribution over classes.
    - opt: Optimizer, selected from torch.optim.
    - sAll: Full stimulus matrix, used for updating model statistics. (nStim x nDim)
    - ctgInd: Vector indicating category of each stimulus row. Used for
            updating statistics (nStim)
    - scheduler: Scheduler for adaptive learning rate, generated with optim.lr_scheduler.
            Default is None.
    - addStimNoise: Boolean indicating whether to add stimulus noise during
            training. Default is True.
    - addRespNoise: Boolean indicating whether to add response noise during
            training. Default is True.
    """
    trainingLoss = np.zeros(nEpochs+1)
    elapsedTime = np.zeros(nEpochs+1)
    # Get the loss of the full dataset stored in the data loader
    trainingLoss[0] = lossFun(model, trainDataLoader.dataset.tensors[0],
            trainDataLoader.dataset.tensors[1]).detach()
    print('Initial loss: ', trainingLoss[0])
    opt.zero_grad()
    # If narrowband, precompute amplitude spectrum for more speed
    if model.filtNorm == 'narrowband':
        sAmp = qm.compute_amplitude_spectrum(s=sAll)
    else:
        sAmp = None
    # Measure time and start loop
    start = time.time()
    for epoch in range(nEpochs):
        for sb, ctgb in trainDataLoader:
            # Generate predictions for batch sb, returned by trainDataLoader 
            model.update_response_statistics(sAll=sAll, ctgInd=ctgInd,
                    sAmp=sAmp)
            loss = lossFun(model, sb, ctgb)     # Compute loss
            loss.backward()                     # Compute gradient
            opt.step()                          # Take one step
            opt.zero_grad()                     # Restart gradient
        # Print model loss
        trainingLoss[epoch+1] = lossFun(model, trainDataLoader.dataset.tensors[0],
                trainDataLoader.dataset.tensors[1]).detach()
        trainingDiff = trainingLoss[epoch+1] - trainingLoss[epoch]
        print('Epoch: %d |  Training loss: %.4f  |  Loss change: %.4f' %
                (epoch+1, trainingLoss[epoch+1], trainingDiff))
        end = time.time()
        elapsedTime[epoch+1] = end - start
        # Apply scheduler step
        if not scheduler == None:
            if "ReduceLROnPlateau" in str(type(scheduler)):
                scheduler.step(trainingLoss[epoch+1])    # adapt learning rate
            else:
                scheduler.step()
    # Do the final response statistics update
    model.update_response_statistics(sAll=sAll, ctgInd=ctgInd, sAmp=sAmp)
    return trainingLoss, elapsedTime


# LOOP TO TRAIN MULTIPLE SEEDS AND CHOOSE BEST
def fit_multiple_seeds(nEpochs, model, trainDataLoader, lossFun, opt_fun,
        sAll, ctgInd,  nSeeds=1, scheduler_fun=None, addStimNoise=True,
        addRespNoise=True, sAmp=None):
    """ Fit AMA model multiple times from different seeds, and keep the result with
    best performance.
    #
    Inputs:
    - nEpochs: Number of epochs for each pair of filters. Integer.
    - model: AMA model object.
    - trainDataLoader: Data loader generated with torch.utils.data.DataLoader.
    - lossFun: Loss function that uses posterior distribution over classes.
    - opt_fun: A function that takes in a model and returns an optimizer.
    - sAll: Full stimulus matrix, used for computing pairwise
            correlations. (nStim x nDim)
    - ctgInd: Vector indicating category of each stimulus row. (nStim)
    - nSeeds: Number of times to train the filters among which to choose
            the best ones. Default is 1.
    - scheduler_fun: Function that takes in an optimizer and returns
            a scheduler for that optimizer. Default is None.
    - addStimNoise: Boolean indicating whether to add stimulus noise during
            training. Default is True.
    - addRespNoise: Boolean indicating whether to add response noise during
            training. Default is True.
    - sAmp: Precomputed amplitude spectrum. Used for speed when the model
            filter normalization is set to "narrowband". (nStim x nDim).
            Optional argument, sAmp is coputed in the function if set to None.
    """
    # If narrowband, precompute amplitude spectrum for more speed
    if model.filtNorm == 'narrowband':
        if sAmp is None:
            sAmp = qm.compute_amplitude_spectrum(s=sAll)
    else:
        sAmp = None
    if (nSeeds>1):
        # Initialize lists to fill
        seedLoss = np.zeros(nSeeds)
        trainingLoss = [None] * nSeeds
        elapsedTimes = [None] * nSeeds
        filters = [None] * nSeeds
        for p in range(nSeeds):
            if (p>0):
                model.reinitialize_trainable(sAll=sAll, ctgInd=ctgInd, sAmp=sAmp)
            # Train the current pair of trainable filters
            opt = opt_fun(model)
            if (scheduler_fun == None):
                scheduler = None
            else:
                scheduler = scheduler_fun(opt)
            # Train random initialization of the model
            trainingLoss[p], elapsedTimes[p] = fit(nEpochs=nEpochs, model=model,
                    trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
                    sAll=sAll, ctgInd=ctgInd, scheduler=scheduler,
                    addStimNoise=addStimNoise, addRespNoise=addRespNoise)
            filters[p] = model.f.detach().clone()
            # Get the final loss of the filters of this seetrainingLossd
            seedLoss[p] = trainingLoss[p][-1]
        # Set the filter with the minimum loss into the model
        minFilt = seedLoss.argmin()
        model.assign_filter_values(filters[minFilt], sAll=sAll, ctgInd=ctgInd,
                sAmp=sAmp)
        # Return only the training loss history of the best filter
        minLoss = trainingLoss[minFilt]
        minElapsed = elapsedTimes[minFilt]
    else:
        opt = opt_fun(model)
        if (scheduler_fun == None):
            scheduler = None
        else:
            scheduler = scheduler_fun(opt)
        minLoss, minElapsed = fit(nEpochs=nEpochs, model=model,
                trainDataLoader=trainDataLoader, lossFun=lossFun, opt=opt,
                sAll=sAll, ctgInd=ctgInd, scheduler=scheduler,
                addStimNoise=addStimNoise, addRespNoise=addRespNoise)
    return minLoss, minElapsed


# TRAIN MODEL FILTERS IN PAIRS, WITH POSSIBLE SEED SELECTION
def fit_by_pairs(nEpochs, model, trainDataLoader, lossFun, opt_fun,
        nPairs, sAll, ctgInd, scheduler_fun=None, seedsByPair=1,
        addStimNoise=True, addRespNoise=True):
    """ Fit AMA model training filters by pairs. After a pair is trained, it
    is fixed in place (no longer trainable), and a new set of trainable
    filters is then initialized and trained. Has the option to try different
    seeds for each pair of filters trained, and choosing the best pair
    #
    Inputs:
    - nEpochs: Number of epochs for each pair of filters. Integer.
    - model: AMA model object.
    - trainDataLoader: Data loader generated with torch.utils.data.DataLoader.
    - lossFun: Loss function that uses posterior distribution over classes.
    - opt_fun: A function that takes in a model and returns an optimizer.
    - nPairs: Number of pairs to train. nPairs=1 corresponds to only training
        the filters included in the input model.
    - sAll: Full stimulus matrix, used for computing pairwise
            correlations. (nStim x nDim)
    - ctgInd: Vector indicating category of each stimulus row. (nStim)
    - seedsByPair: Number of times to train each pair from different random
        initializations, to choose the best pair. Default is 1.
    - scheduler_fun: Function that takes in an optimizer and returns
            a scheduler for that optimizer. Default is None.
    - addStimNoise: Boolean indicating whether to add stimulus noise during
            training. Default is True.
    - addRespNoise: Boolean indicating whether to add response noise during
            training. Default is True.
    """
    if model.filtNorm == 'narrowband':
        sAmp = qm.compute_amplitude_spectrum(s=sAll)
    else:
        sAmp = None
    trainingLoss = [None] * nPairs
    elapsedTimes = [None] * nPairs
    # Measure time and start loop
    start = time.time()
    for p in range(nPairs):
        # If not the first iteration, fix current filters and add new trainable
        if (p>0):
            model.move_trainable_2_fixed(sAll=sAll, ctgInd=ctgInd, sAmp=sAmp)
        print(f'Pair {p}')
        # Train the current pair of trainable filters
        trainingLoss[p], elapsedTimes[p] = fit_multiple_seeds(nEpochs=nEpochs,
                model=model, trainDataLoader=trainDataLoader, lossFun=lossFun,
                opt_fun=opt_fun, sAll=sAll, ctgInd=ctgInd,
                nSeeds=seedsByPair, scheduler_fun=scheduler_fun,
                addStimNoise=addStimNoise, addRespNoise=addRespNoise,
                sAmp=sAmp)
        end = time.time()
        elapsedTime = end - start
        print(f'########## Pair {p+1} trained in {elapsedTime} ##########')
    # Put all the filters into the f model attribute
    fAll = model.fixed_and_trainable_filters().detach().clone()
    model.assign_filter_values(fNew=fAll, sAll=sAll, ctgInd=ctgInd,
            sAmp=sAmp)
    model.add_fixed_filters(fFixed=torch.tensor([]), sAll=sAll,
            ctgInd=ctgInd, sAmp=sAmp)
    return trainingLoss, elapsedTimes


##################################
##################################
#
## LOSS FUNCTIONS
#
##################################
##################################
#
# Define loss functions that take as input AMA model, so
# different outputs can be used with the same fitting functions


def cross_entropy_loss():
    """ Cross entropy loss for AMA.
    model: AMA model object
    s: input stimuli. tensor shaped batch x features
    ctgInd: true categories of stimuli, as a vector with category index
        type torch.LongTensor"""
    def lossFun(model, s, ctgInd):
        posteriors = model.get_posteriors(s, addStimNoise=F)
        nStim = s.shape[0]
        loss = -torch.mean(posteriors[torch.arange(nStim), ctgInd])
        return loss
    return lossFun


def kl_loss():
    """ Negative log-likelihood loss for AMA.
    model: AMA model object
    s: input stimuli. tensor shaped batch x features
    ctgInd: true categories of stimuli, as a vector with category index
        type torch.LongTensor"""
    def lossFun(model, s, ctgInd):
        logProbs = F.log_softmax(model.get_log_likelihood(s), dim=1)
        nStim = s.shape[0]
        loss = -torch.mean(logProbs[torch.arange(nStim), ctgInd])
        return loss
    return lossFun


def mse_loss():
    """ MSE loss for AMA. Computes MSE between the latent variable
    estimate
    model: AMA model object
    s: input stimuli. tensor shaped batch x features
    ctgInd: true categories of stimuli. type torch.LongTensor"""
    mseLoss = torch.nn.MSELoss()
    def lossFun(model, s, ctgInd):
        loss = mseLoss(model.get_estimates(s, method4est='MMSE'),
                model.ctgVal[ctgInd])
        return loss
    return lossFun


def mae_loss():
    """ MAE loss for AMA. Computes MAE between the latent variable
    estimate 
    model: AMA model object
    s: input stimuli. tensor shaped batch x features
    ctgInd: true categories of stimuli. type torch.LongTensor"""
    mseLoss = torch.nn.L1Loss()
    def lossFun(model, s, ctgInd):
        loss = mseLoss(model.get_estimates(s, method4est='MMSE'),
                model.ctgVal[ctgInd])
        return loss
    return lossFun


##################################
##################################
#
## STIMULUS PROCESSING
#
##################################
##################################
#
# Define loss functions that take as input AMA model, so
# different outputs can be used with the same fitting functions


######3 Check this works
def contrast_stim(s, chnl=1):
    """Take a batch of stimuli and convert to Weber contrast stimulus
    That is, subtracts the stimulus mean, and then divides by the mean.
    Arguments:
    s: Stimuli batch. (nStim x nDimensions)
    chnl: Channels into which to separate the stimulus to make each
        channel into contrast individually. """
    s_split = torch.chunk(s, chnl, dim=1)
    s_contrast_split = []
    for s_part in s_split:
        sMean = torch.mean(s_part, axis=1)
        sContrast = torch.einsum('nd,n->nd', (s_part - sMean.unsqueeze(1)), 1/sMean)
        s_contrast_split.append(sContrast)
    sContrast = torch.cat(s_contrast_split, dim=1)
    return sContrast

##################################
##################################
#
## SUMMARIZE MODEL RESULTS
#
##################################
##################################
#
#

# Function that turns posteriors into estimate averages, SDs and CIs
def get_estimate_statistics(estimates, ctgInd, quantiles=[0.025, 0.975]):
    # Compute means and stds for each true level of the latent variable
    estimatesMeans = torch.zeros(ctgInd.max()+1)
    estimatesSD = torch.zeros(ctgInd.max()+1)
    lowCI = torch.zeros(ctgInd.max()+1)
    highCI = torch.zeros(ctgInd.max()+1)
    quantiles = torch.tensor(quantiles)
    for cl in ctgInd.unique():
        levelInd = [i for i, j in enumerate(ctgInd) if j == cl]
        estimatesMeans[cl] = estimates[levelInd].mean()
        estimatesSD[cl] = estimates[levelInd].std()
        (lowCI[cl], highCI[cl]) = torch.quantile(estimates[levelInd], quantiles)
    return {'estimateMean': estimatesMeans, 'estimateSD': estimatesSD,
            'lowCI': lowCI, 'highCI': highCI}


def subsample_covariance(covariance, classInd, filtInd):
    """ Takes a tensor of shape k x d x d holding the covariance
    matrices for k classes and d filters, and returns a smaller
    tensor with the covariances matrices of the classes given in
    classInd, and of the filters in filtInd.
    Eg. if classInd=[2, 3, 4] and filtInd=[0,3], it returns the
    covariance matrix between filters 0 and 3, for classes 2,3,4."""
    covPlt = covariance[classInd, :, :]
    covPlt = covPlt[:, filtInd, :]
    covPlt = covPlt[:, :, filtInd]
    return covPlt


def response_ellipses_subplot(resp, covariance, ctgInd, ctgVal,
        plotFilt=torch.tensor([0,1]), fig=None, ax=None):
    """Do a 2D scatter plot of a set of responses, and draw ellipses to
    show the 2 SD of the Gaussian distribution.
    Inputs:
    - resp: Tensor with filter responses. (nStim x nFilt)
    - covariance: Tensor with covariances for each class. (nClasses x nFilt x nFilt)
    - ctgInd: Class index of each stimulus. (nStim)
    - ctgVal: Vector with the X values of each category.
            Used for the color code
    - plotFilt: Tensor with the indices of the two filters to plot (i.e. the columns
            of resp, and the sub-covariance matrix of covariance)
    - ax: The axis handle on which to draw the ellipse
    """
    # Get the value corresponding to each data point
    respVal = ctgVal[ctgInd]
    # Select relevant covariances
    # Covariances to plot
    covPlt = subsample_covariance(covariance=covariance,
            classInd=ctgInd.unique(), filtInd=plotFilt)
    # Category values associated with the covariances
    covVal = ctgVal[ctgInd.unique()]
    # Plot responses and ellipses
    if ax is None:
        fig, ax = plt.subplots()
        showPlot = True
    else:
        showPlot = False
    # Normalize color plot for shared colors
    norm = colors.Normalize(vmin=ctgVal.min(), vmax=ctgVal.max())
    cmap = cm.get_cmap('viridis')
    sc = ax.scatter(resp[:, plotFilt[0]], resp[:, plotFilt[1]],
            c=respVal, cmap=cmap, s=2, norm=norm, alpha=0.5)
    # create ellipses for each covariance matrix
    for i in range(covPlt.shape[0]):
        cov = covPlt[i, :, :]
        # Get ellipse parameters
        scale, eigVec = torch.linalg.eig(cov)
        scale = torch.sqrt(scale).real
        eigVec = eigVec.real
        # Draw Ellipse
        ell = patches.Ellipse(xy=(0,0), width=scale[0]*4, height=scale[1]*4,
                angle=torch.rad2deg(torch.atan2(eigVec[1, 0], eigVec[0, 0])),
                color=cmap(norm(covVal[i])))
        ell.set_facecolor('none')
        ell.set_linewidth(2)
        ax.add_artist(ell)
    # Label the axes indicating plotted filters
    plt.xlabel(f'Filter {plotFilt[0]}')
    plt.ylabel(f'Filter {plotFilt[1]}')
    if showPlot:
        cax = fig.add_axes([0.90, 0.125, 0.02, 0.755])
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def all_response_ellipses(model, s, ctgInd, ctgStep, colorLabel):
    fig, ax = plt.subplots()
    nPairs = int(model.nFiltAll/2)
    for pp in range(nPairs):
        fPair = torch.tensor([0,1]) + pp*2
        ctgVis = torch.arange(start=0, end=ctgInd.max()+1, step=ctgStep)
        # Extract the stimuli corresponding to these categories
        visInds = torch.where(torch.isin(ctgInd, ctgVis))[0]
        sVis = s[visInds, :]
        ctgIndVis = ctgInd[visInds]
        # Obtain the noisy responses to these stimuli
        respVis = model.get_responses(s=sVis, addStimNoise=True,
                addRespNoise=True)
        respVis = respVis.detach()
        # Plot responses and the ama-estimated ellipses
        ax = plt.subplot(1, nPairs, pp+1)
        response_ellipses_subplot(resp=respVis,
                covariance=model.respCov.detach(),
                ctgInd=ctgIndVis, ctgVal=model.ctgVal,
                plotFilt=fPair,
                fig=fig, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.title(f'Filters {fPair[0]} - {fPair[1]}')
        cax = plt.gca().collections[-1].colorbar
    # Draw color bar and adjust size of plots to avoid overlap
    plt.subplots_adjust(wspace=0.3, right=0.85)
    cmap = cm.get_cmap('viridis')
    norm = colors.Normalize(vmin=model.ctgVal.min(),
            vmax=model.ctgVal.max())
    cax = fig.add_axes([0.91, 0.125, 0.02, 0.755])
    colorbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    colorbar.set_label(colorLabel, labelpad=-70)
    plt.show()



##################################
##################################
#
## PLOT STIMULI AND FILTERS
#
##################################
##################################
#
#

### 1D BINOCULAR IMAGES


def view_1D_bino_image(inputVec, x=[], title=''):
    """
    Plot a vector that contains a 1D binocular image. The plot
    overlaps the first half of the vector (left eye) and second half
    (right eye).
    #
    Arguments:
    - inputVec: Vector to plot. Usually, filter or input image.
    - x: x axis ticks. Optional
    - title: Plot title. Optional
    """
    plt.title(title)
    nPixels = int(max(inputVec.shape)/2)
    if len(x) == 0:
        x = np.arange(nPixels)
    plt.plot(x, inputVec[:nPixels], label='L', color='red')
    plt.plot(x, inputVec[nPixels:], label='R', color='blue')
    plt.ylim(-0.3, 0.3)


def view_all_filters_1D_bino_image(amaPy, x=[]):
    """
    Plot all the filters contained in an ama model, trained to
    process 1D binocular images.
    """
    fAll = amaPy.fixed_and_trainable_filters()
    fAll = fAll.detach()
    nFiltAll = fAll.shape[0]
    nPairs = int(nFiltAll/2)
    for n in range(nFiltAll):
        plt.subplot(nPairs, 2, n+1)
        view_1D_bino_image(fAll[n,:], x=[], title=f'F{n}')


### 1D BINOCULAR VIDEOS


def unvectorize_1D_binocular_video(inputVec, nFrames=15):
    """
    Take a 1D binocular video, in the shape of a vector, and
    reshape it into a 2D matrix, where each row is a time frame,
    each column is a time-changing pixel, and the left and right
    half of the matrix contain the left eye and right eye videos
    respectively.
    #
    Arguments:
    - inputVec: Vector that contains a 1D binocular video. It
        can be  matrix, where each row is a 1D binocular video.
    - frames: Number of time frames in the video
    #
    Outputs:
    - matVideo: 2D format of the 1D video, with rows as frames and
        columns as pixels. (nStim x nFrames x nPixels*2)
    """
    if inputVec.dim() == 1:
        inputVec.unsqueeze(0)
    nVec = inputVec.shape[0]
    nPixels = round(inputVec.shape[1]/(nFrames*2))
    outputMat = torch.zeros(nVec, nFrames, nPixels*2)
    leftEye = inputVec[torch.arange(nVec), 0:(nPixels*nFrames)]
    rightEye = inputVec[torch.arange(nVec), (nPixels*nFrames):]
    leftEye = leftEye.reshape(nVec, nFrames, nPixels)
    rightEye = rightEye.reshape(nVec, nFrames, nPixels)
    outputMat = torch.cat((leftEye, rightEye), dim=2)
    return outputMat


# Plot filters
def view_1D_bino_video(fIn, nFrames=15):
    matFilt = unvectorize_1D_binocular_video(fIn, nFrames=nFrames)
    nFilters = matFilt.shape[0]
    for k in range(nFilters):
        plt.subplot(nFilters, 2, k+1)
        plt.imshow(matFilt[k, :, :].squeeze(), cmap='gray')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    return ax


def view_all_filters_1D_bino_video(amaPy, nFrames=15):
    """
    Plot all the filters contained in an ama model, trained to
    process 1D binocular images.
    """
    fAll = amaPy.fixed_and_trainable_filters()
    fAll = fAll.detach()
    matFilt = unvectorize_1D_binocular_video(fAll, nFrames=nFrames)
    nFiltAll = fAll.shape[0]
    nPairs = int(nFiltAll/2)
    for n in range(nFiltAll):
        plt.subplot(nPairs, 2, n+1)
        view_

