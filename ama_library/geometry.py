import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from matplotlib import patches, colors, cm
import time
import geomstats as gs
import geomstats.geometry.spd_matrices as spd
import geomstats.geometry.symmetric_matrices as sym
import geomstats.geometry.positive_lower_triangular_matrices as chol


def interpolate_covariance_sequence(covariances, nPoints=1,
        metric='BuressWasserstein'):
    if not metric in ['Affine', 'BuressWasserstein', 'LogEuclidean',
                      'Euclidean', 'Cholesky']:
        raise ValueError("Metric provided is not valid")
    if type(covariances) is torch.Tensor:
        covariances = covariances.numpy()
    # Get dimensions of matrices
    nDim = covariances.shape[-1]
    # Initialize the manifold
    manifold = spd.SPDMatrices(n=nDim, equip=False)
    if metric == 'BuressWasserstein':
        manifold.equip_with_metric(spd.SPDBuresWassersteinMetric)
    elif metric == 'Affine':
        manifold.equip_with_metric(spd.SPDAffineMetric)
    elif metric == 'LogEuclidean':
        manifold.equip_with_metric(spd.SPDLogEuclideanMetric)
    elif metric == 'Euclidean':
        manifold.equip_with_metric(spd.SPDEuclideanMetric)
    elif metric == 'Cholesky':
        covariances = np.linalg.cholesky(covariances) # Cholesky decomposition
        manifold = chol.PositiveLowerTriangularMatrices(n=nDim, equip=False)
        manifold.equip_with_metric(chol.CholeskyMetric)
    # Make the geodesics function
    geodesic = manifold.metric.geodesic(initial_point=covariances[:-1,:,:],
            end_point=covariances[1:,:,:])
    # Make parametrizing vector
    t = np.linspace(start=0, stop=1, num=nPoints+2)
    t = t[:-1]
    # Obtain interpolated points in geodesic
    interpolated = geodesic(t)
    # Rearrange the interpolated matrices to the desired order
    newShape = (np.prod(interpolated.shape[0:2]), nDim, nDim)
    interpolated = np.reshape(interpolated, newShape)
    lastCov = np.expand_dims(covariances[-1,:,:], axis=0)
    interpolated = np.concatenate((interpolated, lastCov), axis=0)
    # If used cholesky metric, return back to spdm manifold
    if metric == 'Cholesky':
      interpolated = np.einsum('kim,kjm->kij', interpolated, interpolated)
    return interpolated


def interpolate_means_euclidean(respMean, nPoints):
    nClasses, nFilters = respMean.shape
    respMeanInterp = torch.zeros(((nClasses - 1) * (nPoints) + nClasses, nFilters))
    for i in range(nClasses - 1):
        startClass = respMean[i,:]
        endClass = respMean[i+1,:]
        for j in range(nPoints + 1):
            interpolationRatio = j / (nPoints + 1)
            respMeanInterp[i * (nPoints + 1) + j, :] = (1 - interpolationRatio) * \
              startClass + interpolationRatio * endClass
    respMeanInterp[-1,:] = respMean[-1,:]
    return respMeanInterp





