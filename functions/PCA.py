import numpy as np


def PCA(stan_x, X_mean, num_components, train_evs=None):
    """

    :param stan_x: standardized input values
    :param X_mean: mean of input values
    :param num_components: number of components of test
    :param train_evs: eigenvectors of train points
    :return: returns each test point after dimensionality reduction
    """
    # eigen values
    eigen_values, eigen_vectors = np.linalg.eigh(stan_x)
    # eigenvectors
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    # get num_components eigenvectors
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
    if train_evs is not None:
        eigenvector_subset = train_evs
    X_reduced = np.dot(X_mean, eigenvector_subset)
    return X_reduced, eigenvector_subset
