import numpy as np

# very incomplete still! see data exploration.ipynb for experimentation and splitting.ipynb for some thoughts
# I'm going to update splitting and this file again before our meeting today

def center(X, scale=False):
    mean = np.mean(X, axis=0)
    x_mean = X - mean
    cov_mat = np.cov(x_mean, rowvar=0)
    # c_mat = cov_mat
    if scale:
        corr_mat = np.corrcoef(X_mean, rowvar=False)
        corr_mat[np.isnan(corr_mat)] = cov_mat[np.isnan(corr_mat)]
        # c_mat = corr_mat
    return cov_mat if scale == False else corr_mat


def center_test_values(X, Y):
    mean = np.mean(Y, axis=0)
    X_mean = X - mean
    cov_mat = np.cov(X_mean, axis=0)
    return cov_mat
