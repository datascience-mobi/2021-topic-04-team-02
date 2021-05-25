import numpy as np

def PCA_func(X, num_components):
    # mean
    X_meaned = X - np.mean(X, axis=0)
    # covariance matrix
    cov_mat = np.cov(X_meaned, rowvar=False)
    # eigen values
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    # eigenvectors
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    # subset
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
    return X_reduced