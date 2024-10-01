import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import pycircstat as pcirc
import time

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
    logProbs = F.log_softmax(model.log_likelihoods(s), dim=1)
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


