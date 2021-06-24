import numpy as np


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


def center(X, scale=False):
    """

    :param X: data point out of test set
    :param scale:
    :return: returns either covariance matrix or correlations matrix
    """
    mean = np.mean(X, axis=0)
    X_mean = X - mean
    cov_mat = np.cov(X_mean, rowvar=0)
    # cov_mat = cov_columns(X_mean, X_mean)
    c_mat = cov_mat
    if scale:
        corr_mat = np.corrcoef(X_mean, rowvar=False)
        corr_mat[np.isnan(corr_mat)] = cov_mat[np.isnan(corr_mat)]
        # or replace with 0???
        c_mat = corr_mat
    return c_mat, X_mean


def center_test_values(X, Y, scale=False):
    mean = np.mean(Y, axis=0)
    X_mean = X - mean
    cov_mat = np.cov(X_mean, rowvar=0)
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
