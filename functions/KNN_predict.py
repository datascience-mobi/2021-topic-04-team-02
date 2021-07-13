import numpy as np
from scipy.stats import mode
from scipy.spatial import KDTree


def manhattan_distance(trainvalues_pca, X):
    """

    :param trainvalues_pca: array of training data
    :param X: tested data point
    :return: returns manhattan distance
    """
    distances = np.abs(trainvalues_pca - X).sum(-1)
    return distances


def euclidean_distance(trainvalues_pca, X):
    """

    :param trainvalues_pca: array of training data
    :param X: tested data point
    :return: returns euclidean distance
    """
    distances = np.linalg.norm(trainvalues_pca - X, axis=1)
    return distances

def knn(distance_method_as_string, trainvalues_pca, trainlabels, X, k):
    """

    :param distance_method: euclidean or manhattan as a string
    :param trainvalues_pca: array of training data
    :param trainlabels: labels of the training data
    :param X: tested data point
    :param k: variable k for the knn algorithm
    :return: returns the major vote of the k nearest neighbours based on used distance method distance
    """
    if distance_method_as_string == "euclidean":
        distances = euclidean_distance(trainvalues_pca, X)
    elif distance_method_as_string == "manhattan":
        distances = manhattan_distance(trainvalues_pca, X)
    else:
        print("Distance method not implemented, please use euclidean or manhattan!")
    nearest = trainlabels[np.argsort(distances)[:k]] #extracting the k nearest neighbours from the trainlabels through sorting the distances
    return mode(nearest)[0][0]

def weighted_knn(distance_method_as_string, trainvalues_pca, trainlabels, X, k):
    """

    :param distance_method_as_string: euclidean or manhattan as a string
    :param trainvalues_pca: array of training data
    :param trainlabels: labels of the training data
    :param X: tested data point
    :param k: variable k for the knn algorithm
    :return:
    """
    #function is wrong --> needs to be corrected
    if distance_method_as_string == "euclidean":
        distances = euclidean_distance(trainvalues_pca, X)
    elif distance_method_as_string == "manhattan":
        distances = manhattan_distance(trainvalues_pca, X)
    else:
        print("Distance method not implemented, please use euclidean or manhattan!")
    #weights = 1.0 / distances
    #weights /= weights.sum(axis=0)
    #nearest = trainlabels[np.argsort(distances)[:k]]


    placeholder = weights
    return placeholder

def kdtree_knn(X,k,trainlabels,kdtree,distance_method):
    """

    :param X: tested data point
    :param k: variable k for the knn algorithm
    :param trainlabels: labels of the training data
    :param kdtree: pre calculated kd-tree
    :return:
    """
    if distance_method == "euclidean":
        p = 2
    elif distance_method == "manhattan":
        p = 1
    else:
        print("Distance method not implemented, please use euclidean or manhattan!")
    tree = kdtree
    dd, ii = tree.query(X, k=k,p=p)
    nearest = trainlabels[ii]
    return mode(nearest)[0][0]

def distances_euclidean_testing(trainvalues_pca, trainlabels, testvalues_pca,k):
    """

    :param trainvalues_pca: array of training data
    :param trainlabels: labels of the training data
    :param testvalues_pca: array of testing data
    :param k: variable k for the knn algorithm
    :return:x should return labels of the nearest neighbours for all test data points at once
    """

    distances = np.sqrt(((trainvalues_pca[np.newaxis] - testvalues_pca[:, np.newaxis])**2).sum(axis=-1))  # calculates the euclidean distance between trainvalues and data point X
    nearest = trainlabels[np.argsort(distances)[:,:k]]  # extracting the k nearest neighbours from the trainlabels through sorting the distances
    return nearest