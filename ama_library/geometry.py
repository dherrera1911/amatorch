import numpy as np
import torch
from torch import optim
from matplotlib import patches, colors, cm
import geomstats.backend as gs
import geomstats.geometry.spd_matrices as spd
import geomstats.geometry.positive_lower_triangular_matrices as chol
from geomstats.learning.preprocessing import ToTangentSpace
from scipy.interpolate import CubicSpline


def covariance_interpolation(covariances, nPoints=1,
                             metric='BuressWasserstein',
                             method='geodesic'):
    """ Interpolate between a set of covariance matrices. The interpolation is
      done in the Symmetric Positive Definite
    Matrices manifold.
    -----------------
    Arguments:
    -----------------
    - covariances: Array or tensor containing the covariances, of shape
      (nCtg, nDim, nDim).
    - nPoints: Number of points to interpolate between each pair of
      covariances.
    - metric: Metric to use in the interpolation. Options are:
      - 'Affine': Affine invariant metric.
      - 'BuressWasserstein': Buress-Wasserstein metric.
      - 'LogEuclidean': Log-Euclidean metric.
      - 'Euclidean': Euclidean metric.
      - 'Cholesky': Cholesky decomposition of the covariance matrices.
    - method: Method to use for the interpolation. Options are:
      - 'geodesic': Geodesic interpolation.
      - 'spline': Cubic spline interpolation in the tangent space of the
        Frechet mean.
    -----------------
    Outputs:
    -----------------
    - interpolated: Array containing the interpolated covariances, of shape
        (nCtg * (nPoints + 1), nDim, nDim).
    """
    if not metric in ['Affine', 'BuressWasserstein', 'LogEuclidean',
                      'Euclidean', 'Cholesky']:
        raise ValueError("Metric provided is not valid")
    covariances = gs.array(covariances)
    # Get dimensions of matrices
    nDim = covariances.shape[-1]
    nCovs = covariances.shape[0]
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
        covariances = gs.linalg.cholesky(covariances) # Cholesky decomposition
        manifold = chol.PositiveLowerTriangularMatrices(n=nDim, equip=False)
        manifold.equip_with_metric(chol.CholeskyMetric)
    # Interpolate between the covariances
    if method == 'geodesic':
        # Make the geodesics function
        geodesic = manifold.metric.geodesic(initial_point=covariances[:-1,:,:],
                end_point=covariances[1:,:,:])
        # Make parametrizing vector
        t = gs.linspace(0, 1, nPoints+2)
        t = t[:-1]
        # Obtain interpolated points in geodesic
        interpolated = geodesic(t)
        # Rearrange the interpolated matrices to the desired order
        newShape_dim0 = interpolated.shape[0] * interpolated.shape[1]
        newShape = (newShape_dim0, nDim, nDim)
        interpolated = interpolated.reshape(newShape)
        lastCov = gs.expand_dims(covariances[-1,:,:], 0)
        covariancesInterp = gs.concatenate((interpolated, lastCov), 0)
    elif method == 'spline':
        # Project manifold to the tangent space of the Frechet mean
        tangentMap = ToTangentSpace(manifold)
        tangentVecs = tangentMap.fit_transform(covariances)
        # Fit a cubic spline to the tangent vectors
        tReg = gs.linspace(0, nCovs-1, nCovs) # start, end, steps
        splineInterp = CubicSpline(x=tReg, y=tangentVecs, axis=0)
        # Use the cubic to interpolate the tangent vectors
        t = gs.linspace(0, nCovs-1, (nCovs-1)*(nPoints+1)+1)
        # Obtain interpolated points in the tangent space
        tangentVecsInterp = gs.array(splineInterp(t))
        # Map back to the manifold
        covariancesInterp = tangentMap.inverse_transform(tangentVecsInterp)
    # If used cholesky metric, return back to spdm manifold
    if metric == 'Cholesky':
      covariancesInterp = gs.einsum('kim,kjm->kij', covariancesInterp,
                                    covariancesInterp)
    return covariancesInterp


def interpolate_means(respMean, nPoints, method='geodesic'):
    """ Interpolate between a set of vectors. 
    -----------------
    Arguments:
    -----------------
    - respMean: Array or tensor containing the vectors to interpolate, of
        shape (nCtg, nDim). Interpolation is done over axis=0.
    - nPoints: Number of points to interpolate between each pair of
      covariances.
    - method: Method to use for the interpolation. Options are:
      - 'geodesic': Linear interpolation.
      - 'spline': Cubic spline interpolation
    -----------------
    Outputs:
    -----------------
    - interpolated: Array containing the interpolated vectors, of shape
        (nCtg * (nPoints + 1), nDim).
    """
    nClasses, nFilters = respMean.shape
    respMeanInterp = torch.zeros(((nClasses - 1) * (nPoints) + nClasses, nFilters))
    # convert PyTorch tensor to numpy array
    respMean_np = respMean.detach().numpy()
    if method == 'geodesic':
        for i in range(nClasses - 1):
            startClass = respMean[i,:]
            endClass = respMean[i+1,:]
            for j in range(nPoints + 1):
                interpolationRatio = j / (nPoints + 1)
                respMeanInterp[i * (nPoints + 1) + j, :] = \
                    (1 - interpolationRatio) * startClass + \
                    interpolationRatio * endClass
        # Add the last class that was not added in interpolation
        respMeanInterp[-1, :] = respMean[-1, :]
    elif method == 'spline':
        tReg = np.linspace(start=0, stop=nClasses-1, num=nClasses)
        splineInterp = CubicSpline(x=tReg, y=respMean_np, axis=0)
        t = np.linspace(start=0, stop=nClasses-1, num=(nClasses-1)*(nPoints+1)+1)
        respMeanInterp = splineInterp(t)
    return respMeanInterp


