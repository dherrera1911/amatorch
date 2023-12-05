import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
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
        plotFilt=torch.tensor([0,1]), subsampleFactor=1, fig=None, ax=None,
        lims=1, colorLabel='Category value'):
    """Do a 2D scatter plot of a set of responses, and draw ellipses to
    show the 2 SD of the Gaussian distribution.
    -----------------
    Arguments:
    -----------------
        - covariance: Tensor with covariances for each class. (nClasses x nFilt x nFilt)
        - resp: Tensor with filter responses. (nStim x nFilt)
        - ctgInd: Class index of each stimulus. (nStim)
        - ctgVal: Vector with the X values of each category. Used for
            the color code
        - plotFilt: Tensor with the indices of the two filters to plot (i.e. the columns
                of resp, and the sub-covariance matrix of covariance)
        - subsampleFactor: Factor by which categories are subsampled. Uses subsample
          categories from utilities
        - ax: The axis handle on which to draw the ellipse
        - lims: The limits of the plot. Scalar
    """
    nCtg = covariance.shape[0]
    # If ctgVal not give, make equispaced between -1 and 1
    if ctgVal is None:
        ctgVal = torch.linspace(-1, 1, nCtg)
    # If ctgInd not given, make vector with class indices
    if ctgInd is None:
        ctgInd = torch.arange(covariance.shape[0])
    # Select relevant classes, and subsample vectors
    if nCtg % 2 == 1:
        ctg2plot = au.subsample_categories_centered(nCtg=nCtg,
                                                    subsampleFactor=subsampleFactor)
    else:
        ctg2plot = np.arange(0, nCtg, subsampleFactor)
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
    cmap = sns.diverging_palette(220, 20, s=80, l=70, sep=1, center="dark", as_cmap=True)
    #cmap = cm.get_cmap('twilight_shifted')
    #cmap = sns.color_palette("icefire", as_cmap=True)
    #cmap = sns.color_palette("Spectral", as_cmap=True)
    if not resp is None:
        # Get the value corresponding to each data point
        respVal = ctgVal[ctgInd[np.isin(ctgInd, ctg2plot)]]
        sc = ax.scatter(resp[:, plotFilt[0]], resp[:, plotFilt[1]],
                c=respVal, cmap=cmap, s=40, norm=norm, alpha=0.5)
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
        ell.set_linewidth(3)
        ax.add_artist(ell)
    # Label the axes indicating plotted filters
    # Make the numbers be sub indices
    #plt.xlabel(f'$f_{{{plotFilt[0]+1}}}$ response')
    #plt.ylabel(f'$f_{{{plotFilt[1]+1}}}$ response')
    plt.xlabel(f'f{plotFilt[0]+1} response')
    plt.ylabel(f'f{plotFilt[1]+1} response')
    # Set axis labels font size
    # Set axis limits
    ax.set_xlim(-lims,lims)
    ax.set_ylim(-lims,lims)
    # Set axis ticks
    ticks = [-1, 0, 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # Add color bar
    # To the side
    #cax = fig.add_axes([0.99, 0.13, 0.03, 0.65])
    #cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    #cbar.ax.set_ylabel(colorLabel, rotation=0, ha="right")
    #cbar.ax.yaxis.set_label_coords(7, 1.1)
    cax = fig.add_axes([0.29, 0.90, 0.60, 0.025])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                        orientation="horizontal")
    # Set color bar label above color bar
    # Set color bar label above color bar
    cbar.ax.set_title(colorLabel, loc='center', fontsize=26)
    # Modify tick size
    cbar.ax.tick_params(labelsize=24)
    if showPlot:
        fig.tight_layout(rect=[0, 0, 0.95, 0.92])
        plt.show()


def all_response_ellipses(model, s, ctgInd, ctgStep, colorLabel,
        addStimNoise=True, addRespNoise=True):
    """ Plot the responses and the 95% probability distribution ellipse
    for all pairs of filters in the model.
    -----------------
    Arguments:
    -----------------
      - model: AMA model object. get_responses(), respCov, ctgVal.
      - s: Input stimuli. (nStim x nDim)
      - ctgInd: Category index of each stimulus. (nStim)
      - ctgStep: Integer specifying the step size for the creation of a
        range of category values.
      - colorLabel: A string to use as the label for the color bar in the plots.
      - addStimNoise: Boolean, if true, noise will be added to the stimuli
          to compute the responses
      - addRespNoise: Boolean, if true, noise will be added to the
          model responses.
    """
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
        sizeList=None, maxInd=None, showPlot=True):
    """
    Plot how each element i,j of the covariance matrix changes as a function
    of the level of the latent varaiable.
    -----------------
    Arguments:
    -----------------
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
                # add a bit of noise to prevent overlapping
                x = x + np.random.randn(len(x)) * 0.00005
                # Plot how this element changes as a function of k
                scatter = axs[i, j].scatter(x, elementValues,
                                            color=colors[covIndex],
                                            label=covarianceNames[covIndex],
                                            s=sizeList[covIndex], alpha=1)
                # Draw horizontal line through 0
                axs[i, j].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
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
    if showPlot:
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
    -----------------
    Arguments:
    -----------------
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
    plt.ylim(-0.35, 0.35)


def view_all_filters_1D_bino_image(fAll, x=[]):
    """
    Plot all the filters contained in an ama model, trained to
    process 1D binocular images.
    -----------------
    Arguments:
    -----------------
      - fAll: Matrix that contains all the filters. Each row
          contains a filter.
    """
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
    -----------------
    Arguments:
    -----------------
      - inputVec: Vector that contains a 1D binocular video. It
          can be  matrix, where each row is a 1D binocular video.
      - frames: Number of time frames in the video
    -----------------
    Outputs:
    -----------------
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


def vectorize_2D_binocular_video(matVideo, nFrames=15):
    """
    Inverts the transformation of the unvectorize_1D_binocular_video function.
    Takes the 2D format of the binocular video and converts it back to its 1D form.
    -----------------
    Arguments:
    -----------------
      - matVideo: 2D format of the binocular video (nStim x nFrames x nPixels*2).
      - nFrames: Number of time frames in the video (default: 15).
    -----------------
    Outputs:
    -----------------
      - outputVec: 1D binocular video. It can also be a matrix, where each
          row is a 1D binocular video.
    """
    nStim = matVideo.shape[0]
    nPixels2 = matVideo.shape[2]
    nPixels = nPixels2 // 2  # nPixels for one eye
    # Split the left and right eyes
    leftEye = matVideo[:, :, :nPixels]
    rightEye = matVideo[:, :, nPixels:]
    # Reshape each eye tensor to 1D form
    leftEye = leftEye.reshape(nStim, nPixels * nFrames)
    rightEye = rightEye.reshape(nStim, nPixels * nFrames)
    # Concatenate the left and right eyes along the second dimension (columns)
    outputVec = torch.cat((leftEye, rightEye), dim=1)
    # If there's only one stimulus, we can squeeze to remove the first dimension
    if nStim == 1:
        outputVec = outputVec.squeeze(0)
    return outputVec

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
    -----------------
    Arguments:
    -----------------
      - means: Tensor containing the mean (or median) value of the
          distribution for each x
      - sd: Tensor containing the standard deviation of the distribution
          for each x
      - multiplier: Number of standard deviations to use for the
          confidence interval. Default is 1.96, which corresponds to
          95% confidence interval.
    ----------------
    Outputs:
    ----------------
      - ci: Tensor containing the confidence interval for each x.
    """
    ciHigh = means + multiplier * sd
    ciLow = means - multiplier * sd
    ci = torch.cat((ciLow.unsqueeze(0), ciHigh.unsqueeze(0)), dim=0)
    return ci


def plot_estimate_statistics(estMeans, errorInterval, ctgVal=None,
                              showPlot=True, xLab='Value ', unitsStr='', color='b'):
    """ Plot the estimated mean and confidence interval of the
    model estimates at each value of the latent variable.
    -----------------
    Arguments:
    -----------------
      - estMeans: Tensor containing the mean (or median) value
          of the model estimates for each x
      - errorInterval: Tensor containing the standard deviation of the
          model estimates for each x
      - ctgVal: Tensor containing the value of the latent variable
          for each x. If None, then it is assumed that the latent
          variable is a linearly spaced vector between -1 and 1.
      - showPlot: Boolean indicating whether to show the plot or not.
          Default is True.
      - unitsStr: String indicating the units of the latent variable.
    """
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
    plt.plot(ctgVal, estMeans, color=color, linewidth=4)
    plt.fill_between(ctgVal, errorInterval[0,:], errorInterval[1,:],
                     color=color, alpha=0.2)
    plt.axline((0,0), slope=1, color='black', linestyle='--', linewidth=2)
    plt.xlabel(xLab+unitsStr)
    plt.ylabel('Estimates '+unitsStr)
    if showPlot:
        plt.show()


def plot_posteriors(posteriors, ctgInd=None, ctg2plot=None, ctgVal=None,
                    traces2plot=None, quantiles=[0.16, 0.84], showPlot=True,
                    maxColumns=5):
    """ Plot the individual posteriors obtained from the model, as well as
    the median posterior.
    -----------------
    Arguments:
    -----------------
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
                 linewidth=0.2, alpha=0.3)
        plt.plot(ctgVal, medianPosterior, color='black', linewidth=2)
#        plt.ylim([0, 1])
    if showPlot:
        plt.show()


def plot_posterior_neuron(posteriors, ctgInd=None, ctg2plot=None, ctgVal=None,
                    quantiles=[0.16, 0.84], showPlot=True,
                    maxColumns=5):
    """ Plot the posterior of one of the classes as a function of
    the class of the presented stimulus.
    -----------------
    Arguments:
    -----------------
      - posteriors: Tensor with the posteriors, of size (nStim x nClasses)
      - ctgInd: Category index of each stimulus. (nStim)
      - ctg2plot: Categories to plot. If None, plot all categories.
      - ctgVal: Category values. If None, use interval [-1, 1].
      - showPlot: If True, show the plot. If False, return the plot
    """
    if ctgInd is None:
        ctgInd = torch.zeros(posteriors.shape[0])
    if ctg2plot is None:
        ctg2plot = torch.unique(ctgInd)
    if ctgVal is None:
        ctgVal = torch.linspace(-1, 1, posteriors.shape[1])
    # Get number of columns and rows to subplot
    nColumns = min(len(ctg2plot), maxColumns)
    nRows = int(np.ceil(len(ctg2plot)/nColumns))
    # Compute the mean posterior at the indicated class when the
    # other classes are shown
    ctgIndUninterp = au.reindex_categories(torch.clone(ctgInd))
    ctgIndUnique = torch.unique(ctgInd)
    for i in range(len(ctg2plot)):
        ctgPosteriors = posteriors[:, ctg2plot[i]]
        posteriorStats = au.get_estimate_statistics(ctgPosteriors,
                                                   ctgIndUninterp,
                                                   quantiles=quantiles)
        plt.subplot(nRows, nColumns, i+1)
        # Plot the posteriors
        plt.axvline(x=ctgVal[ctg2plot[i]], color='blue')
        plt.fill_between(ctgVal[ctgIndUnique], posteriorStats['lowCI'],
                         posteriorStats['highCI'], color='black', alpha=0.3)
        plt.plot(ctgVal[ctgIndUnique], posteriorStats['estimateMedian'], color='red')
    if showPlot:
        plt.show()

