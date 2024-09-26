import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches, colors, cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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


def plot_ellipse(mean, cov, ax, color='black'):
    """ Draw an ellipse on the input axis
    -----------------
    Arguments:
    -----------------
      - mean: Tensor with the mean of the Gaussian distribution.
        length 2 tensor
      - cov: Tensor with the covariance matrix of the Gaussian
          distribution. 2x2 tensor
      - ax: Axis handle on which to draw the ellipse.
      - color: Color of the ellipse.
    """
    # Get eigenvalues and eigenvectors
    eigVal, eigVec = torch.linalg.eigh(cov)
    # Get the angle of the ellipse
    angle = torch.atan2(eigVec[1, 0], eigVec[0, 0])
    # Get the length of the semi-axes
    scale = torch.sqrt(eigVal)
    # Plot the ellipse
    ellipse = patches.Ellipse(xy=mean, width=scale[0]*4, height=scale[1]*4,
                              angle=angle*180/np.pi, color=color)
    ellipse.set_facecolor('none')
    ellipse.set_linewidth(3)
    ax.add_patch(ellipse)


def plot_ellipse_set(mean, cov, ax, ctgVal, colorLims=None, colorMap='viridis'):
    """ Plot a set of ellipses, one for each category
    -----------------
    Arguments:
    -----------------
      - mean: Tensor with the means of a set of Gaussian distributions.
        Length nx2 tensor
      - cov: Tensor with the covariance matrix of the Gaussian
          distribution. nx2x2 tensor
      - ctgVal: Vector with the value that determines the color of
          each ellipse in the given colorMap. Length n tensor
      - ax: Axis handle on which to draw the ellipse.
      - colorMap: Color map to use for the ellipses.
    """
    # Get color map
    if isinstance(colorMap, str):
        cmap = plt.get_cmap(colorMap)
    else:
        cmap = colorMap
    # Get number of categories
    nCtg = cov.shape[0]
    # Get color map
    cmap = plt.get_cmap(colorMap)
    # Get the color for each category
    if colorLims is None:
        colorLims = [min(ctgVal), max(ctgVal)]
    norm = Normalize(vmin=colorLims[0], vmax=colorLims[1])
    colors = cmap(norm(ctgVal))
    # Plot each ellipse
    for i in range(nCtg):
        plot_ellipse(mean=mean[i,:], cov=cov[i,:,:], ax=ax, color=colors[i])


def response_scatter(ax, resp, ctgVal, colorLims=None, colorMap='viridis'):
    """ Scatter plot of the responses to the stimuli, with color
    indicating the category value.
    -----------------
    Arguments:
    -----------------
      - ax: Axis handle on which to draw the ellipse.
      - resp: Tensor with the responses to the stimuli. nStim x 2
      - ctgVal: Vector with the color value for each stimulus response.
          Length nStim tensor
      - colorMap: Color map to use for the response points.
    """
    # Get color map
    if isinstance(colorMap, str):
        cmap = plt.get_cmap(colorMap)
    else:
        cmap = colorMap
    # Get the color for each category
    if colorLims is None:
        colorLims = [min(ctgVal), max(ctgVal)]
    norm = Normalize(vmin=colorLims[0], vmax=colorLims[1])
    colors = cmap(norm(ctgVal))
    # Scatter plot
    ax.scatter(resp[:,0], resp[:,1], c=colors, s=40, alpha=0.5)


def add_colorbar(ax, ctgVal, colorLims=None, colorMap='viridis', label='', ticks=None,
                 orientation='vertical', fontSize=24):
    """
    Add a color bar to the axes based on the ctgVal array.
    -----------------
    Arguments:
    -----------------
      - ax: Axis handle on which to draw the ellipse.
      - ctgVal: Array of color values used. Min and max are taken
          from this array.
      - colorMap: Color map to use for the ellipses.
      - label: Label for the color bar.
      - ticks: Specific tick marks to place on the color bar.
    """
    # Get color map
    if isinstance(colorMap, str):
        cmap = plt.get_cmap(colorMap)
    else:
        cmap = colorMap
    # Get fig from ax
    fig = ax.get_figure()
    if colorLims is None:
        colorLims = [min(ctgVal), max(ctgVal)]
    norm = Normalize(vmin=colorLims[0], vmax=colorLims[1])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Determine position of color bar based on orientation
    if orientation == 'horizontal':
        #cax_pos = [0.28, 0.95, 0.60, 0.025]
        cax_pos = [0.28, 1, 0.60, 0.025]
    else:  # vertical
        #cax_pos = [0.96, 0.11, 0.03, 0.65]  # Adjust as needed
        cax_pos = [0.98, 0.11, 0.03, 0.65]  # Adjust as needed
    cax = fig.add_axes(cax_pos)
    cbar = fig.colorbar(sm, cax=cax, ticks=ticks, orientation=orientation)
    cbar.ax.tick_params(labelsize=fontSize)
    cbar.ax.set_title(label, loc='center', fontsize=fontSize)
    cbar.ax.yaxis.set_label_coords(7, 1)


def plot_covariance_values(axes, covariance, xVal=None, color='black',
                           label='', size=None):
    """
    Plot how each element i,j of the covariance matrix changes as a function
    of the level of the latent varaiable.
    -----------------
    Arguments:
    -----------------
      - axes: Axis handle on which to draw the values.
      - covariances: List of arrays of covariance matrices to plot.
          Each element of the list has shape (n, c, c)
          where n is the number of matrices and c is the dimension of each matrix.
          Can be the object respCov that is obtained from an ama object.
      - xVal: List of x axis values for each element in covariances.
      - color: Color of the scatter plot.
      - label: Label for the scatter plot.
      - size: Size of the scatter plot points.
    """
    # Size of the covariance matrices
    c = covariance.shape[1]
    n = covariance.shape[0] # Number of covariance matrices
    if xVal is None:
        xVal = np.linspace(-1, 1, n)
    # Get size of axes
    nAxes = axes.shape
    # Iterate over the lower triangle of the covariance matrices
    for i in range(c):
        for j in range(i+1):
            # Check that there's an ax on which to plot
            if i < nAxes[0] and j < nAxes[1]:
                # Extract the (i,j)-th element from each covariance matrix
                elementValues = covariance[:, i, j]

                # Plot how this element changes as a function of k
                if isinstance(color, str):
                    scatter = axes[i, j].scatter(xVal, elementValues, color=color,
                                                label=label, s=size)
                else:
                    scatter = axes[i, j].scatter(xVal, elementValues, c=xVal,
                                                cmap=color, label=label, s=size)

                # Draw horizontal line through 0
                axes[i, j].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
                # Remove x axis labels if not last row
                if i != c - 1:
                    axes[i, j].set_xticklabels([])
                axes[i, j].set_yticklabels([])
                # Remove redundant plots
                if i != j:
                    axes[j, i].axis('off')
#    # Create the legend in the top right subplot
#    axes[0, -1].legend(legend_handles, covarianceNames, loc="center")
#    axes[0, -1].axis('off')  # turn off axis lines and labels
#    plt.tight_layout()
#    if showPlot:
#        plt.show()


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


def plot_posterior(ax, posteriors, ctgVal=None, traces2plot=None,
                    quantiles=[0.16, 0.84], trueVal=None):
    """ Plot the individual posteriors obtained from the model, as well as
    the median posterior.
    -----------------
    Arguments:
    -----------------
      - ax: Axis handle on which to draw the values.
      - posteriors: Tensor with the posteriors, of size (nStim x nClasses)
      - ctgVal: Category values. If None, use interval [-1, 1].
      - traces2plot: Number of traces to plot per category. If None,
        plot all traces.
    """
    if ctgVal is None:
        ctgVal = torch.linspace(-1, 1, posteriors.shape[1])
    # Compute median posterior
    medianPosterior = np.median(posteriors, axis=0)
    ciPosterior = np.quantile(posteriors, q=quantiles, axis=0)
    if traces2plot is not None:
        tracePosteriors = posteriors[:traces2plot,:]
    else:
        tracePosteriors = posteriors
    # Plot the posteriors
    if trueVal is not None:
        ax.axvline(x=trueVal, color='blue')
    ax.fill_between(ctgVal, ciPosterior[0,:], ciPosterior[1,:],
                      color='black', alpha=0.3)
    ax.plot(ctgVal, np.transpose(tracePosteriors), color='black',
             linewidth=0.05, alpha=0.2)
    ax.plot(ctgVal, medianPosterior, color='black', linewidth=2)


def plot_posterior_neuron(ax, posteriorCtg, ctgInd, ctgVal=None,
                    trueVal=None, quantiles=[0.16, 0.84], meanOrMedian='median'):
    """ Plot the posterior of one of the classes as a function of
    the class of the presented stimulus.
    -----------------
    Arguments:
    -----------------
      - ax: Axis handle on which to draw the values.
      - posteriors: Tensor with the posteriors, of size (nStim x nClasses)
      - ctgInd: Category index of each stimulus. (nStim)
      - ctg2plot: Categories to plot. If None, plot all categories.
      - ctgVal: Category values. If None, use interval [-1, 1].
    """
    if ctgVal is None:
        ctgVal = torch.linspace(-1, 1, np.unique(ctgInd))
    # Compute the mean posterior at the indicated class when the
    # other classes are shown
    posteriorStats = au.get_estimate_statistics(posteriorCtg, ctgInd,
                                               quantiles=quantiles)
    # Plot the true value
    if trueVal is not None:
        ax.axvline(x=trueVal, color='grey', linestyle='--')
    # Sort the ctgVal values and statistics
    ctgValSorted, sortInd = torch.sort(ctgVal)
    # Plot the posteriors
    ax.fill_between(ctgValSorted, posteriorStats['lowCI'][sortInd],
                     posteriorStats['highCI'][sortInd], color='black', alpha=0.2)
    if meanOrMedian == 'mean':
        ax.plot(ctgValSorted, posteriorStats['estimateMean'][sortInd], color='black')
    else:
        ax.plot(ctgValSorted, posteriorStats['estimateMedian'][sortInd], color='black')


