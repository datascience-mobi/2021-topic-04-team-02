import numpy as np
from scipy.stats import mode


def knn_euclidean(trainvalues_pca, trainlabels, X, k):
    """

    :param trainvalues_pca: array of training data
    :param trainlabels: labels of the training data
    :param X: tested data point
    :param k: variable k for the knn algorithm
    :return: returns the major vote of the k nearest neighbours based on euclidean distance
    """
    distances = np.linalg.norm(trainvalues_pca - X, axis=1) #calculates the euclidean distance between trainvalues and data point X
    nearest = trainlabels[np.argsort(distances)[:k]] #extracting the k nearest neighbours from the trainlabels through sorting the distances
    return mode(nearest)[0][0]

def knn_manhattan(trainvalues_pca, trainlabels, X, k):
    """

    :param trainvalues_pca: array of training data
    :param trainlabels: labels of the training data
    :param X: tested data point
    :param k: variable k for the knn algorithm
    :return: returns the major vote of the k nearest neighbours based on manhattan distance
    """
    distances = np.abs(trainvalues_pca - X).sum(-1)
    nearest = trainlabels[np.argsort(distances)[:k]] #extracting the k nearest neighbours from the trainlabels through sorting the distances
    return mode(nearest)[0][0]


def distances_euclidean_testing(trainvalues_pca, trainlabels, testvalues_pca,k):
    """

    :param trainvalues_pca: array of training data
    :param trainlabels: labels of the training data
    :param testvalues_pca: array of testing data
    :param k: variable k for the knn algorithm
    :return:x should return labels of the nearest neughbours for all test data points at once
    """

    distances = np.sqrt(((trainvalues_pca[np.newaxis] - testvalues_pca[:, np.newaxis])**2).sum(axis=-1))  # calculates the euclidean distance between trainvalues and data point X
    nearest = trainlabels[np.argsort(distances)[:,:k]]  # extracting the k nearest neighbours from the trainlabels through sorting the distances
    return nearest