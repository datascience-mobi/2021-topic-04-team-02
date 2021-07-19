import numpy as np


def pca(stan_x, x_mean, num_components, train_evs=None):
    """

    :param stan_x: standardized input values
    :param x_mean: mean of input values
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
    x_reduced = np.dot(x_mean, eigenvector_subset)
    return x_reduced, eigenvector_subset
