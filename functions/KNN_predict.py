import numpy as np
from scipy.stats import mode

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

def knn(distance_method, trainvalues_pca, trainlabels, X, k):
   """

   :param distance_method:
   :param trainvalues_pca: array of training data
   :param trainlabels: labels of the training data
   :param X: tested data point
   :param k: variable k for the knn algorithm
   :return: returns the major vote of the k nearest neighbours based on used distance method distance
   """
   if distance_method == "euclidean":
      distances = euclidean_distance(trainvalues_pca, X)
   elif distance_method == "manhattan":
      distances = manhattan_distance(trainvalues_pca, X)
   else:
      print("Distance method not implemented, please use euclidean or manhattan!")
   nearest = trainlabels[np.argsort(distances)[:k]] #extracting the k nearest neighbours from the trainlabels through sorting the distances
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