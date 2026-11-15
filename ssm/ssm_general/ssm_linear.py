################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
import numpy as np


def calculateSSM(S):
    '''
    Compute snapshot singular value decomposition [Weiss2019_POD].
    A Tutorial on the Proper Orthogonal Decomposition,
    https://doi.org/10.2514/6.2019-3333.

    The snapshot SSM (referred to as POD in paper) is much faster than the
    direct method.

    Parameters
    ----------
    S : (N*dim, M) numpy nd array
        Shape matrix with columns as coordinates of each shape.
        N number of vertices.
        M number of different shapes.
        Possible to stack more than one state of shape into columns of shape matrix.

    Returns
    -------
    S_mean : (N*dim, ) numpy array
        Mean shape matrix with coordinates of mean shape.
    S_modes : (N*dim, M) numpy nd array
        Mode matrix / shape variations / eigenvectors of S.
        M number of different shapes.
    S_var : (N-1, ) numpy array
        Eigenvalues corresponding to S_modes.

    '''
    # calculate mean vector
    ##S_mean_row = S.mean(axis=0)
    S_mean = S.mean(axis=1)

    # subtract mean
    S_dev = S - np.expand_dims(S_mean, axis=1)

    # covariance matrix
    # usually covariance matrix scaled with 1/(len(S[0,:])-1)
    # but for comparibility with direct method use same scale
    # usually : C = np.dot(S_dev.T, S_dev)/(len(S[0,:])-1)
    C = np.dot(S_dev.T, S_dev)/(len(S[0,:])-1)

    # eigenvalues
    S_var, v = np.linalg.eigh(C)

    # sort modes with descending order, highest eigenvalue as first
    idx = S_var.argsort()[::-1]
    S_var = S_var[idx]
    v = v[:, idx]

    # project temporal modes to spatial modes using S_dev
    S_modes = np.matmul(S_dev, v)

    # normalize modes
    for i in range(S_modes.shape[1]):
        S_modes[:, i] = S_modes[:, i]/np.linalg.norm(S_modes[:, i], 2)

    # delete the highest/last mode, since it does not carry any information
    S_modes = S_modes[:,:-1]
    S_var = S_var[:-1]

    return S_mean, S_modes, S_var




def calculateSSM_direct(S):
    '''
    Compute direct singular value decomposition [Weiss2019_POD].
    A Tutorial on the Proper Orthogonal Decomposition,
    https://doi.org/10.2514/6.2019-3333.

    Direct SSM method leads to equal results as the snapshot method, yet the
    computational costs are significantly higher >>> use the snapshot method.

    Parameters
    ----------
    S : (N*dim, M) numpy nd array
        Shape matrix with columns as coordinates of each shape.
        N number of vertices
        M number of different shapes
        Possible to stack more than one state of shape into columns of shape matrix.

    Returns
    -------
    S_mean : (N*dim, ) numpy array
        Mean shape matrix with coordinates of mean shape.
    S_modes : (N*dim, N*dim) numpy nd array
        Mode matrix / shape variations / eigenvectors of S.
        M number of different shapes.
    S_ev : (N*dim, ) numpy array
        Eigenvalues corresponding to S_modes.

    '''
    # direct variant
    S_mean, S_ev = np.empty(len((S))), np.empty(len((S)))

    # calculate mean vector
    S_mean = S.mean(axis=1)

    # subtract mean
    S_dev = S - np.expand_dims(S_mean, axis=1)

    # covariance matrix
    C = np.dot(S_dev, S_dev.T)/(len(S[:, 0])-1)

    # eigenvalues and -vectors
    S_ev, S_modes = np.linalg.eigh(C)

    # sort modes with descending order, highest eigenvalue as first
    idx = S_ev.argsort()[::-1]
    S_ev = S_ev[idx]
    S_modes = S_modes[:, idx]

    return S_mean, S_modes, S_ev
