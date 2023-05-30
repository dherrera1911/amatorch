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
    covPlt = au.subsample_covariance(covariance=covariance,
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



