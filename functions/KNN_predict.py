import numpy as np
from scipy.stats import mode


def knn(trainvalues_pca, trainlabels, X, k):
    """

    :param trainvalues_pca:
    :param trainlabels:
    :param X:
    :param k:
    :return: returns the major vote of the k nearest neighbours
    """
    distances = np.linalg.norm(trainvalues_pca - X, axis=1) #calculates the euclidean distance between trainvalues and data point X
    nearest = trainlabels[np.argsort(distances)[:k]] #extracting the k nearest neighbours from the trainlabels through sorting the distances
    return mode(nearest)[0][0]


def knn_manhattan(trainvalues_pca, trainlabels, testvalues_pca, k):
    """

    :param trainvalues_pca:
    :param trainlabels:
    :param X:
    :param k:
    :return:
    """
    distances = np.abs(trainvalues_pca[np.newaxis] - testvalues_pca[:, np.newaxis]).sum(axis=-1)
    nearest = trainlabels[np.argsort(distances)[:,:k]]

    return mode(nearest)[0][0]

def knn_euclidean(trainvalues_pca, trainlabels, testvalues_pca, k):
    """

    :param trainvalues_pca:
    :param trainlabels:
    :param X:
    :param k:
    :return:x
    """
    distances = np.sqrt(((trainvalues_pca[np.newaxis] - testvalues_pca[:, np.newaxis])**2).sum(axis=-1))
    nearest = trainlabels[np.argsort(distances)[:,:k]]

    return mode(nearest)[0][0]