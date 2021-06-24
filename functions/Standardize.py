import numpy as np


def cov_mod(m, n, rowvar=True):
    """
    Gives the covariance matrix of m using the mean values of n.
    Derived from np.cov.
    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables.
    n : array_like
        An array containing multiple variables and observations.
    rowvar : bool, optional
        If `rowvar` is True (default), then arrays transposed so each row represents a
        variable, with observations in the columns. If false then not transposed.
    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.
    """
    # Data type: double for higher precision
    dtype = np.result_type(m, np.float64)

    # Transpose matrices so that entries of each variable (pixel position) are in rows instead of columns
    X = np.array(m, dtype=dtype)
    X2 = np.array(n, dtype=dtype)
    if rowvar:
        X = X.T
        X2 = X2.T

    # Averages taken from second (training) array
    avg = np.mean(X2, axis=1)
    # unweighted, axis=1 means mean calculated over rows

    # N-1 calculated:
    N_1 = X.shape[1] - 1

    # Averages subtracted:
    X = X - avg[:, None]
    # None adds another dimension so that subtraction is row-wise

    # Skalarprodukt
    X_T = X.T
    c = np.dot(X, X_T)

    # Multiplied by 1/(N-1)
    c = c * np.true_divide(1, N_1)
    # Squeeze removes unnecessary dimensions
    return c.squeeze()


def center(X, Y, scale=False):
    if Y == "None":
        # there must be a nicer way to do this
        mean = np.mean(X, axis=0)
        Y = X
    else:
        mean = np.mean(Y, axis=0)
    X_mean = X - mean
    c_mat = cov_mod(X, Y)
    if scale:
        d = np.diag(c_mat)
        std_dev = np.sqrt(d)
        c_mat = c_mat / std_dev[:, None]
        c_mat = c_mat / std_dev[None, :]
        # missing solution for NaN? 0?!
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
