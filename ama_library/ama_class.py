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
        # If no initial filters, initialize filters and set length to 1
        fInit = torch.randn(nFilt, sAll.shape[1])
        fInit = F.normalize(fInit, p=2, dim=1)
        self.f = nn.Parameter(fInit)    # Model parameters
        geotorch.sphere(self, "f")
        # Get the dimensions of different relevant vectors
        self.nFilt = self.f.shape[0]
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
        self.noiseCov = torch.eye(self.nFilt).repeat(self.nClasses,1,1) \
                * self.filterSigma
        self.respCovs = torch.einsum('fd,jdb,gb->jfg', self.f,
                self.stimCovs, self.f)
        self.respCovs = self.respCovs + self.noiseCov
        self.respMeans = torch.einsum('fd,jd->jf', self.f, self.stimMeans)

    def update_response_statistics(self):
        """ Update (in place) the conditional response means and covariances to match the
        current object filters """
        # Response covariances, size nClasses*nFilt*nFilt
        self.respCovs = torch.einsum('fd,jdb,gb->jfg', self.f,
                self.stimCovs, self.f) # Get the response covariances.
        self.respCovs = self.respCovs + self.noiseCov # Add image & neural variability
        # Response means, size nClasses*nFilt
        self.respMeans = torch.einsum('fd,jd->jf', self.f, self.stimMeans)

    def get_posteriors(self, s):
        """ Compute the class posteriors for each stimulus in s.
        Input: s (nPoints x nDim) is stimulus matrix
        Output: posteriors (nPoints x nClasses). Each row has the posterior
        probability distribution over all classes (each column corresponds to a class)"""
        # If given only vector as input, add singleton dimension
        if s.dim() == 2:
            nPoints = s.shape[0]
        else:
            nPoints = 1
            s = s.unsqueeze(0)
        ###########
        # 1) Responses of the filters to the stimuli. size nPoints*nFilt
        resp = torch.einsum('fd,nd->fn', self.f, s)
        # 2) Difference between responses and class means. size nPoints*nFilt*nClasses
        respDiff = resp.unsqueeze(2).repeat(1,1,self.nClasses) - \
                self.respMeans.unsqueeze(1).repeat(1,nPoints,1).transpose(0,2)
        ## Get the log-likelihood of each class
        # 3) Quadratic component of log-likelihood (with negative sign)
        quadratics = -0.5*torch.einsum('dnj,jdc,cnj->nj', respDiff,
                self.respCovs.inverse(), respDiff)
        # 4) Constant term of log-likelihood
        llConst = -0.5 * self.nFilt * torch.log(2*torch.tensor(torch.pi)) - \
            0.5 * torch.logdet(self.respCovs)
        # 5) Add quadratics and constants and softmax to get posterior probs
        posteriors = F.softmax(quadratics + llConst.repeat(nPoints, 1), dim=1)
        return posteriors

    def get_estimates(self, s, method4est='MAP'):
        """ Compute latent variable estimates for each stimulus in s.
        Input: s (nPoints x nDim) is stimulus matrix
        Output: estimates (nPoints). Vector with the estimate for each stimulus """
        posteriors = self.get_posteriors(s)
        if method4est=='MAP':
            # Get maximum posteriors indices of each stim, and its value
            (a, estimateInd) = torch.max(posteriors, dim=1)
            estimates = self.ctgVal[estimateInd]
        elif method4est=='MMSE':
            estimates = torch.einsum('nj,j->n', posteriors, self.ctgVal)
        return estimates

    def assign_filter_values(self, fNew):
        self.f = nn.Parameter(fNew) # Model parameters
        self.nFilt = self.f.shape[0]
        self.noiseCov = torch.eye(self.nFilt).repeat(self.nClasses,1,1) \
                * self.filterSigma
        self.update_response_statistics()

    def add_new_filters(self, nFiltNew=2):
        """ Add new, random filters to the filters already contained in the model.
        nFiltNew: number of new fiters to add"""
        # If no initial filters, initialize filters and set length to 1
        fNew = F.normalize(torch.randn(nFiltNew, self.nDim), p=2, dim=1)
        fOld = self.f.detach().clone()
        fAll = torch.cat((fOld, fNew))  # Model parameters
        parametrize.remove_parametrizations(self, "f", leave_parametrized=True)
        self.assign_filter_values(fAll.clone())
        geotorch.sphere(self, "f")
        self.f = fAll

