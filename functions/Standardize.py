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
    x = np.array(m, dtype=dtype)
    x2 = np.array(n, dtype=dtype)
    if rowvar:
        x = x.T
        x2 = x2.T

    # Averages taken from second (training) array
    avg = np.mean(x2, axis=1)
    # axis=1 means mean calculated over rows

    # N-1 calculated:
    n_1 = x.shape[1] - 1

    # Averages subtracted:
    x = x - avg[:, None]
    # None adds another dimension so that subtraction is row-wise

    # Scalar product
    x_t = x.T
    c = np.dot(x, x_t)

    # Multiplied by 1/(N-1)
    c = c * np.true_divide(1, n_1)
    # Squeeze removes unnecessary dimensions created by None
    return c.squeeze()


def center(x, y: np.ndarray = None, scale=False):
    """

    :param x: Array of data to be centered and scaled
    :param y: Array with which X is standardized
    :param scale: If False (default) covariance matrix as output. If True then correlation matrix.
    :return: Covariance matrix or correlation matrix of X. And centered matrix: X-mean.
    """
    if y is None:
        mean = np.mean(x, axis=0)
        y = x
    else:
        mean = np.mean(y, axis=0)
    x_mean = x - mean
    c_mat = cov_mod(x, y)
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
        x_mean = x_mean[:, ~np.any(x_mean == 0, axis=0)]
    return c_mat, x_mean
