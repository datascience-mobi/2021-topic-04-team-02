import numpy as np


def standardize(data, z=True):
    """

    :param data: array of data
    :param z:
    :return: row wise aka image wise z transformation
    """
    # Calculate mean of each image
    mean = np.mean(data, axis=1)

    # Subtract mean from each pixel in image and divide by image standard deviation
    data_mod = data.T - mean
    if z:
        data_mod = data_mod / np.std(data, axis=1)
    return data_mod.T


def cov_mod(m, n, rowvar=True):
    """

    :param m: Array where each row of `m` represents a variable, and each column a single
        observation.
    :param n: Array with same number of variables as m.
    :param rowvar: If True, then arrays transposed so each row represents a
        variable, with observations in the columns.
    :return: Gives the covariance matrix of m using the mean values of n.
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
    # axis=1 means mean calculated over rows

    # N-1 calculated:
    N_1 = X.shape[1] - 1

    # Averages subtracted:
    X = X - avg[:, None]
    # None adds another dimension so that subtraction is row-wise

    # Scalar product
    X_T = X.T
    c = np.dot(X, X_T)

    # Multiplied by 1/(N-1)
    c = c * np.true_divide(1, N_1)
    # Squeeze removes unnecessary dimensions created by None
    return c.squeeze()


def center(X, Y: np.ndarray = None, scale=False):
    """

    :param X: Array of data to be centered and scaled
    :param Y: Array with which X is standardized
    :param scale: If False (default) covariance matrix as output. If True then correlation matrix.
    :return: Covariance matrix or correlation matrix of X. And centered matrix: X-mean.
    """
    if Y is None:
        mean = np.mean(X, axis=0)
        Y = X
    else:
        mean = np.mean(Y, axis=0)
    X_mean = X - mean
    c_mat = cov_mod(X, Y)
    if scale:
        # correlation matrix calculated:
        # diagonal entries of covariance matrix are the variance
        d = np.diag(c_mat)
        # square root gives the standard deviation
        std_dev = np.sqrt(d)

        # ignore error message of dividing zero by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            # row-wise then column-wise division by standard deviation
            c_mat = c_mat / std_dev[:, None]
            c_mat = c_mat / std_dev[None, :]

        # Columns with only missing values amputated
        c_mat = c_mat[~np.isnan(c_mat).all(axis=1), :]
        # Because matrix is symmetric corresponding rows also removed
        c_mat = c_mat[:, ~np.isnan(c_mat).any(axis=0)]
        # Amputated variables also removed from X-mean matrix
        X_mean = X_mean[:, ~np.any(X_mean == 0, axis=0)]
    return c_mat, X_mean

