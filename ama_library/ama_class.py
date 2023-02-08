import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import geotorch
import torch.nn.utils.parametrize as parametrize

# Define model class
class AMA(nn.Module):
    def __init__(self, sAll, ctgInd, nFilt=2, filterSigma=0.02, ctgVal=None):
        """ AMA model object.
        sAll: input stimuli. shape batch x features
        ctgInd: category index of each stimulus
        nFilt: number of filters to train (optional if fInit not None)
        filterSigma: variance of filter response noise
        fInit: user defined filters (optional). shape nFilt x features """
        super().__init__()
        # Make initial random filters
        fInit = torch.randn(nFilt, sAll.shape[1])
        fInit = F.normalize(fInit, p=2, dim=1)
        self.f = nn.Parameter(fInit)    # Model parameters
        self.fFixed = torch.tensor([])  # Attribute with fixed (non-trainable) filters. Start as empty
        geotorch.sphere(self, "f")
        # Get the dimensions of different relevant vectors
        self.nFilt = self.f.shape[0]
        self.nFiltAll = self.nFilt      # Number of filters including fixed filters
        self.nDim = self.f.shape[1]
        self.nClasses = np.unique(ctgInd).size
        # If no category values given, assign equispaced values in [-1,1]
        if ctgVal == None:
            ctgVal = np.linspace(-1, 1, self.nClasses)
        self.ctgVal = ctgVal
        # Compute the conditional statistics of the stimuli
        self.stimCovs = torch.zeros(self.nClasses, self.nDim, self.nDim)
        self.stimMeans = torch.zeros(self.nClasses, self.nDim)
        for cl in range(self.nClasses):
            levelInd = [i for i, j in enumerate(ctgInd) if j == cl]
            sLevel = sAll[levelInd, :]
            self.stimCovs[cl, :, :] = torch.cov(sLevel.transpose(0,1))
            self.stimMeans[cl, :] = torch.mean(sLevel, 0)
        # Get conditional response statistics
        self.filterSigma = filterSigma
        self.noiseCov = torch.eye(self.nFiltAll).repeat(self.nClasses,1,1) \
                * self.filterSigma
        self.respCovs = torch.einsum('fd,jdb,gb->jfg', self.f,
                self.stimCovs, self.f)
        self.respCovs = self.respCovs + self.noiseCov
        self.respMeans = torch.einsum('fd,jd->jf', self.f, self.stimMeans)

    def fixed_and_trainable_filters(self):
        """ Return a tensor with all the filters (fixed and trainable)"""
        return torch.cat((self.fFixed, self.f))

    def update_response_statistics(self):
        """ Update (in place) the conditional response means and covariances
        to match the current object filters """
        self.nFiltAll =  self.fFixed.shape[0] + self.f.shape[0]
        fAll = self.fixed_and_trainable_filters() # Get all filters (fixed and trainable)
        self.noiseCov = torch.eye(self.nFiltAll).repeat(self.nClasses,1,1) \
                * self.filterSigma
        # Update covariances, size nClasses*nFilt*nFilt
        self.respCovs = torch.einsum('fd,jdb,gb->jfg', fAll,
                self.stimCovs, fAll) # Get the response covariances.
        self.respCovs = self.respCovs + self.noiseCov # Add image & neural variability
        # Update means, size nClasses*nFilt
        self.respMeans = torch.einsum('fd,jd->jf', fAll, self.stimMeans)

    def get_posteriors(self, s):
        """ Compute the class posteriors for each stimulus in s.
        Input: s (nPoints x nDim) is stimulus matrix
        Output: posteriors (nPoints x nClasses). Each row has the posterior
        probability distribution over all classes for a stimulus (each column
        corresponds to a class)"""
        # If given only vector as input, add singleton dimension
        if s.dim() == 2:
            nPoints = s.shape[0]
        else:
            nPoints = 1
            s = s.unsqueeze(0)
        ###########
        fAll = self.fixed_and_trainable_filters() # Append fixed and trainable filters together
        # 1) Responses of the filters to the stimuli. size nPoints*nFilt
        resp = torch.einsum('fd,nd->fn', fAll, s)
        # 2) Difference between responses and class means. size nPoints*nFilt*nClasses
        respDiff = resp.unsqueeze(2).repeat(1,1,self.nClasses) - \
                self.respMeans.unsqueeze(1).repeat(1,nPoints,1).transpose(0,2)
        ## Get the log-likelihood of each class
        # 3) Quadratic component of log-likelihood (with negative sign)
        quadratics = -0.5*torch.einsum('dnj,jdc,cnj->nj', respDiff,
                self.respCovs.inverse(), respDiff)
        # 4) Constant term of log-likelihood
        llConst = -0.5 * self.nFiltAll * torch.log(2*torch.tensor(torch.pi)) - \
            0.5 * torch.logdet(self.respCovs)
        # 5) Add quadratics and constants and softmax to get posterior probs
        posteriors = F.softmax(quadratics + llConst.repeat(nPoints, 1), dim=1)
        return posteriors

    def get_estimates(self, s, method4est='MAP'):
        """ Compute latent variable estimates for each stimulus in s.
        Input: s (nPoints x nDim) is stimulus matrix
        Output: estimates (nPoints). Vector with the estimate for each stimulus """
        posteriors = self.get_posteriors(s).double()
        if method4est=='MAP':
            # Get maximum posteriors indices of each stim, and its value
            (a, estimateInd) = torch.max(posteriors, dim=1)
            estimates = self.ctgVal[estimateInd]
        elif method4est=='MMSE':
            estimates = torch.einsum('nj,j->n', posteriors, self.ctgVal)
        return estimates

    def assign_filter_values(self, fNew):
        """ Assign new values to the model filters.
        Updates model parameter accordingly.
        Input: fNew (nFilt x nDim) is the matrix with the new filters as rows.
            The new number of filters doesn't need to match the old number."""
        parametrize.remove_parametrizations(self, "f", leave_parametrized=True)
        # Model parameters. Important to clone fNew, because apparently geotorch modifies it
        self.f = nn.Parameter(fNew.clone()) 
        geotorch.sphere(self, "f")
        self.f = fNew
        self.nFilt = self.f.shape[0]
        self.nFiltAll = self.nFilt + self.fFixed.shape[0]
        self.update_response_statistics()

    def add_new_filters(self, nFiltNew=2):
        """ Add new, random filters to the filters already contained in the model.
        nFiltNew: number of new fiters to add"""
        # Initialize new random filters and set length to 1 
        fNew = F.normalize(torch.randn(nFiltNew, self.nDim), p=2, dim=1)
        fOld = self.f.detach().clone()
        f = torch.cat((fOld, fNew))  # Concatenate old and new filters
        self.assign_filter_values(f)

    def add_fixed_filters(self, fFixed):
        """ Add new filters to the model, that are not trainable parameters.
        Input: fFixed (nFilt x nDim) is the tensor with the new filters as rows."""
        self.fFixed = fFixed
        self.update_response_statistics()

    def reinitialize_trainable(self):
        """ Re-initialize the trainable filters to random values """
        fRandom = torch.randn(self.nFilt, self.nDim)
        fRandom = F.normalize(fRandom, p=2, dim=1)
        self.assign_filter_values(fRandom)

