import numpy as np


def cov_mod(m, n, rowvar=True):
    """
    Estimate a covariance matrix, given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.
    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    n : array_like
        An array containing multiple variables and observations.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.
    """

    dtype = np.result_type(m, np.float64)

    X = np.array(m, ndmin=2, dtype=dtype)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return np.array([]).reshape(0, 0)
    dtype2 = np.result_type(n, np.float64)
    X2 = np.array(n, ndmin=2, dtype=dtype2)
    if not rowvar and X2.shape[0] != 1:
        X2 = X2.T
    if X2.shape[0] == 0:
        return np.array([]).reshape(0, 0)

    # Averages to be subtracted taken from second (training) array
    avg = np.average(X2, axis=1)

    # N-1 calculated:
    N_1 = X.shape[1] - 1

    # Averages subtracted:
    X -= avg[:, None]

    # [...]
    X_T = X.T
    c = np.dot(X, X_T.conj())

    # Multiplied by 1/(N-1)
    c *= np.true_divide(1, N_1)
    # output squeeze [...]
    return c.squeeze()


def center(X, Y, scale=False):
    if Y == "None":
        mean = np.mean(X, axis=0)
        Y = X
    else:
        mean = np.mean(Y, axis=0)
    X_mean = X - mean
    cov_mat = cov_mod(X, Y, rowvar=False)
    c_mat = cov_mat
    if scale:
        # WIP
        corr_mat = np.corrcoef(X_mean, rowvar=False)
        corr_mat[np.isnan(corr_mat)] = cov_mat[np.isnan(corr_mat)]
        c_mat = corr_mat
    return c_mat, X_mean


# Sad, inefficient and unused code below:

def corr_columns(mat):
    x = len(mat[0])
    c = (1 / x)
    cor_mat = np.zeros((x, x))
    for i in range(0, x):
        for j in range(0, x):
            cor_mat[i, j] = c * np.dot(((mat[:, i] - np.mean(mat[:, i])) / np.std(mat[:, i])),
                                       ((mat[:, j] - np.mean(mat[:, j])) / np.std(mat[:, j])))

    return cor_mat


def cov_columns(mat, mat2):
    """
    calculates the covariance matrix
    :param mat: Test data or training data
    :param mat2: Training data
    """
    x = len(mat2[0])
    c = (1 / (x - 1))
    # Which should we use here? n-1 or n? np.cov uses n-1
    cov_mat = np.zeros((x, x))
    for i in range(0, x):

        for j in range(0, x):
            cov_mat[i, j] = c * np.dot((mat[:, i] - np.mean(mat2[:, i])), (mat[:, j] - np.mean(mat2[:, j])))

    return cov_mat


def center_test_values(X, Y, scale=False):
    mean = np.mean(Y, axis=0)
    X_mean = X - mean
    cov_mat = np.cov(X_mean, rowvar=False)
    # cov_mat = cov_columns(X_mean, Y)
    c_mat = cov_mat
    if scale:
        x = len(Y[0])
        c = (1/x)
        corr_mat = np.zeros((x, x))
        for i in range(0, x):
            for j in range(0, x):
                corr_mat[i, j] = c * np.dot(((X[:, i] - np.mean(Y[:, i])) / np.std(Y[:, i])),
                                            ((X[:, j] - np.mean(Y[:, j])) / np.std(Y[:, j])))
        corr_mat[np.isnan(corr_mat)] = cov_mat[np.isnan(corr_mat)]
        c_mat = corr_mat
    return c_mat, X_mean
