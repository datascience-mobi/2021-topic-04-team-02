import numpy as np


def PCA_func(X, num_components, train_mean = None, train_evs = None):
    # mean
    mean = np.mean(X, axis =0)
    # if we want to use the PCA that has been trained on the train set for the test set,
    # we require the same "shift" of the mean, therefore, we use the train_mean on the test set
    if train_mean is not None:
        mean = train_mean
    X_mean = X - mean
    cov_mat = np.cov(X_mean,rowvar=0)
    # eigen values
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    # eigenvectors
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    # subset
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
    # if we want to use the PCA that has been trained on the train set for the test set,
    # we want to use the eigenvector subset that has been calculated on the train set for classification
    # on the test set:
    if train_evs is not None:
        eigenvector_subset = train_evs
    X_reduced = np.dot(X_mean, eigenvector_subset)
    # we return X_reduced (the input reduced to num_components PCA),
    # the mean and the eigenvector subset (if we run the function for training, we need these for testing)
    return X_reduced, mean, eigenvector_subset