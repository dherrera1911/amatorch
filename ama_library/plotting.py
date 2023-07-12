import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import patches, colors, cm
from ama_library import quadratic_moments as qm
from ama_library import utilities as au
import time


##################################
##################################
#
## PLOT COVARIANCE MATRICES
#
##################################
##################################
#
#


def response_ellipses_subplot(covariance, resp=None, ctgInd=None, ctgVal=None,
        plotFilt=torch.tensor([0,1]), subsampleFactor=1, fig=None, ax=None):
    """Do a 2D scatter plot of a set of responses, and draw ellipses to
    show the 2 SD of the Gaussian distribution.
    #
    Arguments:
        - covariance: Tensor with covariances for each class. (nClasses x nFilt x nFilt)
        - resp: Tensor with filter responses. (nStim x nFilt)
        - ctgInd: Class index of each stimulus. (nStim)
        - ctgVal: Vector with the X values of each category. Used for
            the color code
        - plotFilt: Tensor with the indices of the two filters to plot (i.e. the columns
                of resp, and the sub-covariance matrix of covariance)
        - subsampleFactor: Factor by which categories are subsampled
        - ax: The axis handle on which to draw the ellipse
    """
    nCtg = covariance.shape[0]
    # If ctgVal not give, make equispaced between -1 and 1
    if ctgVal is None:
        ctgVal = torch.linspace(-1, 1, nCtg)
    # If ctgInd not given, make vector with class indices
    if ctgInd is None:
        ctgInd = torch.arange(covariance.shape[0])
    # Select relevant classes, and subsample vectors
    ctg2plot = au.subsample_categories(nCtg=nCtg, subsampleFactor=subsampleFactor)
    if not resp is None:
        resp = resp[np.isin(ctgInd, ctg2plot),:]
    # Select relevant covariances
    # Covariances to plot
    covPlt = au.subsample_covariance(covariance=covariance, classInd=ctg2plot,
            filtInd=plotFilt)
    # Category values associated with the covariances
    covVal = ctgVal[ctg2plot]
    # Plt responses and ellipses
    if ax is None:
        fig, ax = plt.subplots()
        showPlot = True
    else:
        showPlot = False
    # Normalize color plot for shared colors
    norm = colors.Normalize(vmin=ctgVal.min(), vmax=ctgVal.max())
    cmap = cm.get_cmap('viridis')
    if not resp is None:
        # Get the value corresponding to each data point
        respVal = ctgVal[ctgInd[np.isin(ctgInd, ctg2plot)]]
        sc = ax.scatter(resp[:, plotFilt[0]], resp[:, plotFilt[1]],
                c=respVal, cmap=cmap, s=5, norm=norm, alpha=0.5)
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
    plt.xlabel(f'Filter {plotFilt[0]+1}')
    plt.ylabel(f'Filter {plotFilt[1]+1}')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    #cax = fig.add_axes([0.90, 0.125, 0.02, 0.755])
    #plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    if showPlot:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def all_response_ellipses(model, s, ctgInd, ctgStep, colorLabel,
        addStimNoise=True, addRespNoise=True):
    fig, ax = plt.subplots()
    nPairs = int(model.nFiltAll/2)
    if nPairs > 2:
        nRows = 2
        nCols = int(np.ceil(nPairs/2))
    else:
        nRows = 1
        nCols = nPairs
    for pp in range(nPairs):
        fPair = torch.tensor([0,1]) + pp*2
        ctgVis = torch.arange(start=0, end=ctgInd.max()+1, step=ctgStep)
        # Extract the stimuli corresponding to these categories
        visInds = torch.where(torch.isin(ctgInd, ctgVis))[0]
        sVis = s[visInds, :]
        ctgIndVis = ctgInd[visInds]
        # Obtain the noisy responses to these stimuli
        respVis = model.get_responses(s=sVis, addStimNoise=True,
                addRespNoise=addRespNoise)
        respVis = respVis.detach()
        # Plot responses and the ama-estimated ellipses
        ax = plt.subplot(nRows, nCols, pp+1)
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


def plot_covariance_values(covariances, xVal=None, covarianceNames=None,
        sizeList=None, maxInd=None):
    """
    Plot how each element i,j of the covariance matrix changes as a function
    of the level of the latent varaiable.
    #
    Arguments:
    - covariances: List of arrays of covariance matrices to plot.
        Each element of the list has shape (n, c, c)
        where n is the number of matrices and c is the dimension of each matrix.
        Can be the object respCov that is obtained from an ama object.
    - covarianceNames: List of labels for each element in covariances.
    - xVal: List of x axis values for each element in covariances.
    """
    # Checking if covarianceNames is not given then assign default names
    if covarianceNames is None:
        covarianceNames = ["Covariance " + str(i) for i in range(len(covariances))]
    if sizeList is None:
        sizeList = np.ones(len(covariances)) * 0.5
    # Generate colors for each element in covariances
    colors = plt.cm.rainbow(np.linspace(0, 1, len(covariances)))
    # Size of the covariance matrices
    if maxInd is None:
      c = covariances[0].shape[1]
    else:
      c = maxInd
    # Create a grid of subplots with c rows and c columns
    fig, axs = plt.subplots(c, c, figsize=(15, 15))
    # Handles for storing legend items
    legend_handles = []
    # Iterate over the covariances list
    for covIndex, covariance in enumerate(covariances):
        n = covariance.shape[0] # Number of covariance matrices
        if xVal is None:
            x = np.linspace(-1, 1, n)
        else:
            x = xVal[covIndex]
        # Iterate over the lower triangle of the covariance matrices
        for i in range(c):
            for j in range(i+1):
                # Extract the (i,j)-th element from each covariance matrix
                elementValues = covariance[:, i, j]
                x = x + np.random.randn(len(x)) * 0.00005 # add a bit of noise to prevent overlapping
                # Plot how this element changes as a function of k
                scatter = axs[i, j].scatter(x, elementValues, color=colors[covIndex],
                        label=covarianceNames[covIndex], s=sizeList[covIndex], alpha=1)
                # Add scatter plot to legend handles on first pass
                if i == 0 and j == 0:
                    legend_handles.append(scatter)
                # Remove x axis labels if not last row
                if i != c - 1:
                    axs[i, j].set_xticklabels([])
                axs[i, j].set_yticklabels([])
                # Remove redundant plots
                if i != j:
                    axs[j, i].axis('off')
    # Create the legend in the top right subplot
    axs[0, -1].legend(legend_handles, covarianceNames, loc="center")
    axs[0, -1].axis('off')  # turn off axis lines and labels
    plt.tight_layout()
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

###########
### 1D BINOCULAR IMAGES
###########


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

###########
### 1D BINOCULAR VIDEOS
###########

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
        inputVec = inputVec.unsqueeze(0)
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
    nPairs = int(np.ceil(nFilters/2))
    for k in range(nFilters):
        plt.subplot(nPairs, 2, k+1)
        plt.imshow(matFilt[k, :, :].squeeze(), cmap='gray')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    return ax


def view_filters_list(filtersList, common_title, num_array, nFrames):
    if len(filtersList) != len(num_array):
        raise ValueError("Length of filtersList and num_array must be the same")
    nCols = len(filtersList)
    for i, fIn in enumerate(filtersList):
        matFilt = unvectorize_1D_binocular_video(fIn, nFrames=nFrames)
        nFilters = matFilt.shape[0]
        for k in range(nFilters):
            plt.subplot(nFilters, nCols, k * nCols + i + 1)
            plt.imshow(matFilt[k, :, :].squeeze(), cmap='gray')
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if k == 0:
                formatted_num = "{:.0e}".format(num_array[i])
                plt.title(f"{common_title} {formatted_num}", fontsize=8)
    plt.show()


def plot_loss_list(lossList):
    numPairs = len(lossList)
    for i in range(numPairs):
        plt.subplot(1, numPairs, i+1)
        plt.plot(np.log(lossList[i]))
    plt.show()



##################################
##################################
#
## PLOT MODEL OUTPUTS
#
##################################
##################################
#
#

def sd_to_ci(means, sd, multiplier=1.96):
    """
    Convert standard deviation to confidence interval.
    """
    ciHigh = means + multiplier * sd
    ciLow = means - multiplier * sd
    ci = torch.cat((ciLow.unsqueeze(0), ciHigh.unsqueeze(0)), dim=0)
    return ci


def plot_estimate_statistics_sd(estMeans, errorInterval, ctgVal=None,
                                showPlot=True, unitsStr=''):
    if ctgVal is None:
        ctgVal = torch.linspace(-1, 1, len(estMeans))
    if not torch.is_tensor(ctgVal):
        ctgVal = torch.tensor(ctgVal)
    if errorInterval.dim() == 1:
        errorInterval = sd_to_ci(means=estMeans, sd=errorInterval,
                                 multiplier=1)
    # convert to numpy for matplotlib compatibility
    estMeans = estMeans.detach().numpy()
    ctgVal = ctgVal.detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(ctgVal, estMeans, color='blue')
    plt.fill_between(ctgVal, errorInterval[0,:], errorInterval[1,:],
                     color='blue', alpha=0.2)
    plt.axline((0,0), slope=1, color='black')
    plt.xlabel('Value '+unitsStr)
    plt.ylabel('Estimates '+unitsStr)
    if showPlot:
        plt.show()


def plot_posteriors(posteriors, ctgInd=None, ctg2plot=None, ctgVal=None,
                    traces2plot=None, quantiles=[0.16, 0.84], showPlot=True):
    """ Plot the individual posteriors obtained from the model, as well as
    the median posterior.
    Arguments:
      - posteriors: Tensor with the posteriors, of size (nStim x nClasses)
      - ctgInd: Category index of each stimulus. (nStim)
      - ctg2plot: Categories to plot. If None, plot all categories.
      - ctgVal: Category values. If None, use interval [-1, 1].
      - traces2plot: Number of traces to plot per category. If None,
        plot all traces.
      - showPlot: If True, show the plot. If False, return the plot
    """
    if ctgInd is None:
        ctgInd = torch.zeros(posteriors.shape[0])
    if ctg2plot is None:
        ctg2plot = torch.unique(ctgInd)
    if ctgVal is None:
        ctgVal = torch.linspace(-1, 1, posteriors.shape[1])
    # Get number of columns and rows to subplot
    maxColumns = 5
    nColumns = min(len(ctg2plot), maxColumns)
    nRows = int(np.ceil(len(ctg2plot)/nColumns))
    # Remove categories not to plot
    inds2keep = torch.isin(ctgInd, torch.tensor(ctg2plot))
    posteriors = posteriors[inds2keep, :]
    ctgInd = ctgInd[inds2keep]
    # Compute median posterior for each class
    for i in range(len(ctg2plot)):
        (iInds,) = torch.where(ctgInd == ctg2plot[i])
        ctgPosteriors = posteriors[iInds, :].numpy()
        medianPosterior = np.median(ctgPosteriors, axis=0)
        ciPosterior = np.quantile(ctgPosteriors, q=quantiles, axis=0)
        if traces2plot is not None:
            tracePosteriors = ctgPosteriors[:traces2plot,:]
        else:
            tracePosteriors = ctgPosteriors
        # Plot the posteriors
        plt.subplot(nRows, nColumns, i+1)
        plt.axvline(x=ctgVal[ctg2plot[i]], color='blue')
        plt.fill_between(ctgVal, ciPosterior[0,:], ciPosterior[1,:],
                          color='black', alpha=0.3)
        plt.plot(ctgVal, tracePosteriors.transpose(), color='red',
                 linewidth=0.3, alpha=0.3)
        plt.plot(ctgVal, medianPosterior, color='black', linewidth=2)
    if showPlot:
        plt.show()


