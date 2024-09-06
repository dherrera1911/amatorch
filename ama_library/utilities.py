import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from scipy.stats import circmean, circstd
import pycircstat as pcirc
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
# the filters in pairs, or from multiple seeds


# Define loop function to train the model
def fit(nEpochs, model, trainDataLoader, lossFun, opt, scheduler=None,
        sTst=None, ctgIndTst=None, printProg=True):
    """
    Fit AMA model using the posterior distribuions generated by the model.
    ----------------
    Arguments:
    ----------------
      - nEpochs: Number of epochs. Integer.
      - model: AMA model object.
      - trainDataLoader: Data loader generated with torch.utils.data.DataLoader.
      - lossFun: Loss function to evaluate.
      - opt: Optimizer, selected from torch.optim.
      - scheduler: Scheduler for adaptive learning rate, generated with
              optim.lr_scheduler. Default is None.
      - sTst: Test stimulus matrix, used for computing test
              loss. (nStim x nDim). Default is None.
      - ctgIndTst: Vector indicating category of each test stimulus row.
              Used for computing test loss. (nStim). Default is None.
    ----------------
    Outputs:
    ----------------
      - trnLoss: Vector of training loss at each epoch. (nEpochs)
      - tstLoss: Vector of test loss at each epoch
      - elapsedTime: Vector of elapsed time at each epoch. (nEpochs)
    """
    trnLoss = np.zeros(nEpochs+1)
    tstLoss = np.zeros(nEpochs+1)
    elapsedTime = np.zeros(nEpochs+1)
    # Get the loss of the full dataset stored in the data loader
    trnLoss[0] = lossFun(model=model, s=trainDataLoader.dataset.tensors[0],
                              ctgInd=trainDataLoader.dataset.tensors[1]).detach()
    if not sTst == None:
        tstLoss[0] = lossFun(model=model, s=sTst, ctgInd=ctgIndTst)
    print(f"Init Train loss: {trnLoss[0]:.4f}  | "
          f"Test loss: {tstLoss[0]:.4f}")
    opt.zero_grad()
    # TAKE THE TIME AND START LOOP
    start = time.time()
    if printProg:
        # Print headers
        print("-"*72)
        print(f"{'Epoch':^5} | {'Train loss':^12} | {'Diff (e-3)':^10} | "
              f"{'Test loss':^12} | {'Diff (e-3)':^10} | {'Time (s)':^8}")
        print("-"*72)
    for epoch in range(nEpochs):
        ### MAIN TRAINING LOOP
        for sb, ctgb in trainDataLoader:
            # Update model statistics to the new filters
            model.update_response_statistics()
            loss = lossFun(model, sb, ctgb)     # Compute loss
            loss.backward()                     # Compute gradient
            opt.step()                          # Take one step
            opt.zero_grad()                     # Restart gradient
        ### PRINT MODEL LOSS
        trnLoss[epoch+1] = lossFun(
            model=model,
            s=trainDataLoader.dataset.tensors[0],
            ctgInd=trainDataLoader.dataset.tensors[1]).detach()
        trainingDiff = trnLoss[epoch+1] - trnLoss[epoch]
        if not sTst == None:
            tstLoss[epoch+1] = lossFun(model=model, s=sTst, ctgInd=ctgIndTst)
        tstDiff = tstLoss[epoch+1] - tstLoss[epoch]
        # Print progress
        if printProg:
            print(f"{epoch+1:^5} | "
                  f"{trnLoss[epoch]:>12.3f} | "
                  f"{trainingDiff*1000:>10.1f} | "
                  f"{tstLoss[epoch]:>12.3f} | "
                  f"{tstDiff*1000:>10.1f} | "
                  f"{elapsedTime[epoch]:^8.1f}")
        end = time.time()
        elapsedTime[epoch+1] = end - start
        # Apply scheduler step
        if type(scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step(trnLoss[epoch+1])    # adapt learning rate
        elif type(scheduler) == optim.lr_scheduler.StepLR:
            scheduler.step()
    print("")
    # DO THE FINAL RESPONSE STATISTICS UPDATE
    model.update_response_statistics()
    return trnLoss, tstLoss, elapsedTime


# LOOP TO TRAIN MULTIPLE SEEDS AND CHOOSE BEST
def fit_multiple_seeds(nEpochs, model, trainDataLoader, lossFun, opt_fun,
        nSeeds=1, scheduler_fun=None, sTst=None, ctgIndTst=None,
        printProg=False):
    """
    Fit AMA model multiple times from different seeds, and keep the result with
    best performance.
    ----------------
    Arguments:
    ----------------
      - nEpochs: Number of epochs for each pair of filters. Integer.
      - model: AMA model object.
      - trainDataLoader: Data loader generated with torch.utils.data.DataLoader.
      - lossFun: Loss function that uses posterior distribution over classes.
      - opt_fun: A function that takes in a model and returns an optimizer.
      - nSeeds: Number of times to train the filters among which to choose
              the best ones. Default is 1.
      - scheduler_fun: Function that takes in an optimizer and returns
              a scheduler for that optimizer. Default is None.
      - sTst: Test stimulus matrix, used for computing test
              loss. (nStim x nDim). Default is None.
      - ctgIndTst: Vector indicating category of each test stimulus row.
              Used for computing test loss. (nStim). Default is None.
    ----------------
    Outputs:
    ----------------
      - trnLoss: Numpy array with training loss for each seed, with rows
              sorted in increasing order of final loss (nSeeds x nEpochs).
      - tstLoss: Numpy array with test loss for each seed, with rows
              sorted in increasing order of final loss (nSeeds x nEpochs).
      - elapsedTime: Numpy array with elapsed time for each seed, with
              rows sorted in increasing order of final loss (nSeeds x nEpochs).
      - filters: List of filters for each seed, sorted in increasing
              order of final loss (nSeeds x nDim).
    """
    # INITIALIZE LISTS TO FILL WITH TRAINING PROGRESS INFORMATION
    seedLoss = np.zeros(nSeeds)
    trnLoss = np.zeros((nSeeds, nEpochs+1))
    tstLoss = np.zeros((nSeeds, nEpochs+1))
    elapsedTimes = np.zeros((nSeeds, nEpochs+1))
    filters = [None] * nSeeds
    # LOOP OVER SEEDS
    for p in range(nSeeds):
        print(f'##########      SEED {p+1}      ########## \n ')
        # If not first seed, reinitialize the model
        if (p>0):
            model.reinitialize_trainable()
            model.update_response_statistics()
        # Set up optimizer and scheduler
        opt = opt_fun(model)
        if (scheduler_fun == None):
            scheduler = None
        else:
            scheduler = scheduler_fun(opt)
        # TRAIN MODEL WITH THIS SEED
        trnLoss[p,:], tstLoss[p,:], elapsedTimes[p,:] = fit(
            nEpochs=nEpochs, model=model, trainDataLoader=trainDataLoader,
            lossFun=lossFun, opt=opt, scheduler=scheduler, sTst=sTst,
            ctgIndTst=ctgIndTst, printProg=printProg)
        # Save filters
        filters[p] = model.f.detach().clone()
        # Get the loss for these filters
        if not sTst is None:
            seedLoss[p] = tstLoss[p,-1]
        else:
            seedLoss[p] = trnLoss[p,-1]
    # Put best filter into the model
    minFilt = seedLoss.argmin()
    model.assign_filter_values(fNew=filters[minFilt])
    model.update_response_statistics()
    # Sort outputs by increasing loss
    trnLoss = trnLoss[seedLoss.argsort(),:]
    tstLoss = tstLoss[seedLoss.argsort(),:]
    elapsedTimes = elapsedTimes[seedLoss.argsort(),:]
    filters = [filters[i] for i in seedLoss.argsort()]
    return trnLoss, tstLoss, elapsedTimes, filters


# TRAIN MODEL FILTERS IN PAIRS, WITH POSSIBLE SEED SELECTION
def fit_by_pairs(nEpochs, model, trainDataLoader, lossFun, opt_fun,
        nPairs, scheduler_fun=None, seedsByPair=1, sTst=None, ctgIndTst=None,
        printProg=False):
    """
    Fit AMA model training filters by pairs. After a pair is trained, it
    is fixed in place (no longer trainable), and a new set of trainable
    filters is then initialized and trained. Has the option to try different
    seeds for each pair of filters trained, and choosing the best pair
    ----------------
    Arguments:
    ----------------
      - nEpochs: Number of epochs for each pair of filters. Integer.
      - model: AMA model object.
      - trainDataLoader: Data loader generated with torch.utils.data.DataLoader.
      - lossFun: Loss function that uses posterior distribution over classes.
      - opt_fun: A function that takes in a model and returns an optimizer.
      - nPairs: Number of pairs to train. nPairs=1 corresponds to only training
          the filters included in the input model.
      - seedsByPair: Number of times to train each pair from different random
          initializations, to choose the best pair. Default is 1.
      - scheduler_fun: Function that takes in an optimizer and returns
              a scheduler for that optimizer. Default is None.
      - sTst: Test stimulus matrix, used for computing test
              loss. (nStim x nDim). Default is None.
      - ctgIndTst: Vector indicating category of each test stimulus row.
              Used for computing test loss. (nStim). Default is None.
    ----------------
    Outputs:
    ----------------
      - trnLoss: List with training loss for each pair of filters trained.
              list of length nPairs, each element is a tensor with size
              (seedsByPair x nEpochs)
      - tstLoss: List with test loss for each pair of filters trained.
              list of length nPairs, each element is a tensor with size
              (seedsByPair x nEpochs)
      - elapsedTimes: List with elapsed times for each pair of filters trained.
              list of length nPairs, each element is a tensor with size
              (seedsByPair x nEpochs)
      - filters: List of different seed filters for each pair trained.
              list of length nPairs, where each element is a list of length
              seedByPair, containing a tensor with the filters trained at
              that step, of size (2 x nDim)
    """
    trnLoss = [None] * nPairs
    tstLoss = [None] * nPairs
    elapsedTimes = [None] * nPairs
    filters = [None] * nPairs
    # Measure time and start loop
    start = time.time()
    for p in range(nPairs):
        # If not the first iteration, fix current filters and add new trainable
        if (p>0):
            model.move_trainable_2_fixed()
            model.update_response_statistics()
        print("#"*45)
        print(f'##########      FILTER PAIR {p+1}      ##########')
        print("#"*45, "\n ")
        # Train the current pair of trainable filters
        trnLoss[p], tstLoss[p], elapsedTimes[p], filters[p] = \
                fit_multiple_seeds(
                    nEpochs=nEpochs, model=model, trainDataLoader=trainDataLoader,
                    lossFun=lossFun, opt_fun=opt_fun, nSeeds=seedsByPair,
                    scheduler_fun=scheduler_fun, sTst=sTst, ctgIndTst=ctgIndTst,
                    printProg=printProg)
        end = time.time()
        elapsedTime = end - start
        minutes, seconds = divmod(int(elapsedTime), 60)
        print(f'########## PAIR {p+1} TRAINED IN {minutes:02d}:{seconds:02d} '
              '########## \n ')
    # Put all the filters into f
    fAll = model.all_filters().detach().clone()
    model.assign_filter_values(fAll)
    model.add_fixed_filters(fFixed=torch.tensor([]))
    model.update_response_statistics()
    return trnLoss, tstLoss, elapsedTimes, filters


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


def cross_entropy_loss(model, s, ctgInd):
    """
    Cross entropy loss. (negative posterior of true class)
    ----------------
    Arguments:
    ----------------
      - model: AMA model object
      - s: input stimuli. tensor shaped batch x features
      - ctgInd: true categories of stimuli. Vector with category index
        type torch.LongTensor
    ----------------
    Outputs:
    ----------------
      - loss: Cross entropy loss
    """
    posteriors = model.get_posteriors(s)
    nStim = s.shape[0]
    loss = -torch.mean(torch.log(posteriors[torch.arange(nStim), ctgInd]))
    return loss


def kl_loss(model, s, ctgInd):
    """
    ----------------
    Arguments:
    ----------------
      - model: AMA model object
      - s: input stimuli. tensor shaped batch x features
      - ctgInd: true categories of stimuli, as a vector with category index
    ----------------
    Outputs:
    ----------------
      - loss: Negative LL loss
    """
    logProbs = F.log_softmax(model.get_ll(s), dim=1)
    nStim = s.shape[0]
    correctClassLogProbs = logProbs[torch.arange(nStim), ctgInd]
    loss = -torch.mean(correctClassLogProbs)
    return loss


def mse_loss():
    """
    ----------------
    Arguments:
    ----------------
      - model: AMA model object
      - s: input stimuli. tensor shaped batch x features
      - ctgInd: true categories of stimuli, as a vector with category index
    ----------------
    Outputs:
    ----------------
      - loss: MSE loss
    """
    mseLoss = torch.nn.MSELoss()
    loss = mseLoss(model.get_estimates(s, method4est='MMSE'),
            model.ctgVal[ctgInd])
    return loss


def mae_loss():
    """
    ----------------
    Arguments:
    ----------------
      - model: AMA model object
      - s: input stimuli. tensor shaped batch x features
      - ctgInd: true categories of stimuli, as a vector with category index
    ----------------
    Outputs:
    ----------------
      - loss: MAE loss
    """
    mseLoss = torch.nn.L1Loss()
    loss = mseLoss(model.get_estimates(s, method4est='MMSE'),
            model.ctgVal[ctgInd])
    return loss


##################################
##################################
#
## STIMULUS PROCESSING
#
##################################
##################################


def noise_total_2_noise_pix(sigmaEqv, numPix):
    """ Calculate the level of noise (i.e. standar deviation) to implement
    decision-variable noise with standard deviation given by sigmaEqv.
    ----------------
    Arguments:
    ----------------
      - sigmaEqv: Standard deviation of decision variable
      - numPix: Number of pixels in the stimulus
    ----------------
    Outputs:
    ----------------
      - sigmaPix: Standard deviation of pixel-level noise
    """
    sigmaPix = sigmaEqv * np.sqrt(numPix)
    return sigmaPix


def normalize_stimuli_channels(s, nChannels=1):
    """ Normalize the stimuli in s to unit norm. If nChannels > 1,
    the columns of s are separated into nChannels that are normalized
    separately.
    ----------------
    Arguments:
    ----------------
      - s: Stimuli to normalize. (nStim x nDim)
      - nChannels: Number of channels into which to divide stimuli.
    ----------------
    Outputs:
    ----------------
      - sNormalized: Normalized stimuli. (nStim x nDim)
    """
    n, d = s.shape
    # Reshape s to have an extra dimension for the groups
    sReshaped = s.view(n, nChannels, -1)
    # Calculate the norms for each group separately
    group_norms = torch.norm(sReshaped, dim=2, keepdim=True)
    # Normalize the groups separately
    sNormalized = sReshaped / group_norms
    # Compute the overall normalization factor
    normFactor = torch.sqrt(torch.tensor(nChannels))
    # Multiply each row by the normalization factor to make the whole row have unit length
    sNormalized = (sNormalized.view(n, -1) / normFactor).view(n, d)
    return sNormalized


def category_means(s, ctgInd):
    """ Compute the mean of the stimuli for each category
    ----------------
    Arguments:
    ----------------
      - s: Stimuli. (nStim x nDim)
      - ctgInd: Category index for each stimulus. (nStim x 1)
    ----------------
    Outputs:
    ----------------
      - stimMean: Mean of the stimuli for each category. (nCtg x nDim)
    """
    nDim = int(s.shape[1])
    # Compute the mean of the stimuli for each category
    device = s.device
    nClasses = torch.unique(ctgInd).size()[0]
    stimMean = torch.zeros(nClasses, nDim, device=device)
    for cl in range(nClasses):
        mask = (ctgInd == cl)
        sClass = s[mask]
        stimMean[cl,:] = torch.mean(sClass, dim=0)
    return stimMean


def category_secondM(s, ctgInd):
    """ Compute the second moment of the stimuli for each category
    ----------------
    Arguments:
    ----------------
      - s: Stimuli. (nStim x nDim)
      - ctgInd: Category index for each stimulus. (nStim x 1)
    ----------------
    Outputs:
    ----------------
      - stimSM: Second moment of the stimuli for each category. (nCtg x nDim x nDim)
    """
    nDim = int(s.shape[1])
    # Compute the second moment of the stimuli for each category
    nClasses = torch.unique(ctgInd).size()[0]
    device = s.device
    stimSM = torch.zeros(nClasses, nDim, nDim, device=device)
    for cl in range(nClasses):
        mask = (ctgInd == cl)
        sClass = s[mask]
        nStimLevel = sClass.size(0)
        stimSM[cl,:,:] = torch.einsum('nd,nb->db', sClass, sClass) / nStimLevel
    return stimSM


def secondM_2_cov(secondM, mean):
    """Convert matrices of second moments to covariances, by
    subtracting the product of the mean with itself.
    ----------------
    Arguments:
    ----------------
      - secondM: Second moment matrix. E.g. computed with
           'isotropic_ctg_resp_secondM'. (nClasses x nFilt x nFilt,
           or nFilt x nFilt)
      - mean: mean matrix. E.g. computed with 'isotropic_ctg_resp_mean'.
           (nClasses x nFilt, or nFilt)
    ----------------
    Outputs:
    ----------------
      - covariance: Covariance matrices. (nClasses x nFilt x nFilt)
    """
    if secondM.dim() == 2:
        secondM = secondM.unsqueeze(0)
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)
    # Get the multiplying factor to make covariance unbiased
    covariance = secondM - torch.einsum('cd,cb->cdb', mean, mean)
    return covariance


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
def get_estimate_statistics(estimates, ctgInd, quantiles=torch.tensor([0.16, 0.84])):
    """ Compute the mean, standard deviation and confidence intervals
    of the estimates for each level of the latent variable.
    ----------------
    Arguments:
    ----------------
      - estimates: Estimates of the latent variable for each stimulus.
        (nStim x 1)
      - ctgInd: Category index for each stimulus. (nStim x 1)
      - quantiles: Quantiles to use for the confidence intervals.
    ----------------
    Outputs:
    ----------------
      - statsDict: Dictionary with the mean, standard deviation and
        confidence intervals for each level of the latent variable.
    """
    # Compute means and stds for each true level of the latent variable
    estimatesMeans = torch.zeros(ctgInd.max()+1)
    estimatesMedians = torch.zeros(ctgInd.max()+1)
    estimatesSD = torch.zeros(ctgInd.max()+1)
    lowCI = torch.zeros(ctgInd.max()+1)
    highCI = torch.zeros(ctgInd.max()+1)
    quantiles = torch.as_tensor(quantiles)
    for cl in ctgInd.unique():
        mask = (ctgInd == cl)
        estLevel = estimates[mask]  # Stimuli of the same category
        estimatesMeans[cl] = estLevel.mean()
        estimatesMedians[cl] = torch.median(estLevel)
        estimatesSD[cl] = estLevel.std()
        (lowCI[cl], highCI[cl]) = torch.quantile(estLevel, quantiles)
    statsDict = {'estimateMean': estimatesMeans,
                 'estimateMedian': estimatesMedians,
                 'estimateSD': estimatesSD,
                 'lowCI': lowCI, 'highCI': highCI}
    return statsDict


def get_estimate_circular_statistics(estimates, ctgInd, quantiles=[0.16, 0.84]):
    """ Compute the circular mean, standard deviation and confidence intervals
    of the estimates for each level of the latent variable.
    ----------------
    Arguments:
    ----------------
      - estimates: Estimates of the latent variable for each stimulus.
        shape (nStim x 1).
      - ctgInd: Category index for each stimulus. shape (nStim x 1).
      - quantiles: Quantiles to use for the confidence intervals.
    ----------------
    Outputs:
    ----------------
      - statsDict: Dictionary with the mean, standard deviation and
        confidence intervals for each level of the latent variable.
    """
    estimatesMeans = np.zeros(ctgInd.max()+1)
    estimatesMedians = np.zeros(ctgInd.max()+1)
    estimatesSD = np.zeros(ctgInd.max()+1)
    lowCIMean = np.zeros(ctgInd.max()+1)
    highCIMean = np.zeros(ctgInd.max()+1)
    lowCIMedian = np.zeros(ctgInd.max()+1)
    highCIMedian = np.zeros(ctgInd.max()+1)
    for cl in np.unique(ctgInd):
        mask = (ctgInd == cl)
        estLevel = np.deg2rad(estimates[mask])  # Stimuli of the same category
        # If estLev has even number of values, remove one, for pcirc to work right
        if len(estLevel) % 2 == 0:
            estLevel = estLevel[:-1]
        # Compute mean and median
        estimatesMeans[cl] = pcirc.mean(estLevel)
        estimatesMedians[cl] = pcirc.median(estLevel)
        # Compute SD
        estimatesSD[cl] = pcirc.std(estLevel)
        # Compute difference to the median
        circDiff = pcirc.cdiff(np.ones(len(estLevel))*estimatesMedians[cl],
                               estLevel)
        # Compute the quantiles of the differences to the median
        lowCIMedian[cl] = estimatesMedians[cl] + np.percentile(circDiff, quantiles[0]*100)
        highCIMedian[cl] = estimatesMedians[cl] + np.percentile(circDiff, quantiles[1]*100)
        # Compute difference to the mean
        circDiff = pcirc.cdiff(np.ones(len(estLevel))*estimatesMeans[cl], estLevel)
        # Compute the quantiles of the differences to the mean
        lowCIMean[cl] = estimatesMeans[cl] + np.percentile(circDiff, quantiles[0]*100)
        highCIMean[cl] = estimatesMeans[cl] + np.percentile(circDiff, quantiles[1]*100)
    # Convert radiants to degrees
    statsDict = {
        'estimateMean': torch.rad2deg(torch.tensor(estimatesMeans, dtype=torch.float32)),
        'estimateMedian': torch.rad2deg(torch.tensor(estimatesMedians, dtype=torch.float32)),
        'estimateSD': torch.rad2deg(torch.tensor(estimatesSD, dtype=torch.float32)),
        'lowCIMedian': torch.rad2deg(torch.tensor(lowCIMedian, dtype=torch.float32)),
        'highCIMedian': torch.rad2deg(torch.tensor(highCIMedian, dtype=torch.float32)),
        'lowCIMean': torch.rad2deg(torch.tensor(lowCIMean, dtype=torch.float32)),
        'highCIMean': torch.rad2deg(torch.tensor(highCIMean, dtype=torch.float32))
    }
    return statsDict


##################################
##################################
#
## UTILITY FUNCTIONS
#
##################################
##################################
#
#


def repeat_stimuli(s, ctgInd, nReps):
    """ Repeat the stimuli in s and the category indices in ctgInd
    nReps times.
    ----------------
    Arguments:
    ----------------
      - s: Stimuli. (nStim x nDim)
      - ctgInd: Category index for each stimulus. (nStim x 1)
      - nReps: Number of times to repeat the stimuli.
    ----------------
    Outputs:
    ----------------
      - sAll: Repeated stimuli. (nStim*nReps x nDim)
      - ctgIndAll: Repeated category index for each stimulus.
          (nStim*nReps x 1)
    """
    sRep = s.repeat(nReps, 1, 1)
    sRep = sRep.transpose(0, 1).reshape(-1, s.shape[1])
    ctgIndRep = ctgInd.repeat_interleave(nReps)
    return sRep, ctgIndRep


def sort_categories(ctgVal, ctgInd):
    """ Sort the categories by their values, and reindex the category
    indices accordingly.
    ----------------
    Arguments:
    ----------------
      - ctgVal: Values of the latent variable for each category. (nCtg)
      - ctgInd: Category index for each stimulus. (nStim)
    ----------------
    Outputs:
    ----------------
      - ctgValSorted: Sorted values of the latent variable for each category. (nCtg)
      - ctgIndSorted: Sorted category index for each stimulus. (nStim)
    """
    # Rearrange the values of ctgVal and of ctgInd so that
    # ctgVal is in ascending order
    sortedValInds = torch.argsort(ctgVal)
    ctgValSorted = ctgVal[sortedValInds]
    # Update ctgInd to match the new ordering of ctgVal
    _, ctgIndNew = torch.sort(sortedValInds)
    ctgIndSorted = ctgIndNew[ctgInd]
    return ctgValSorted, ctgIndSorted


