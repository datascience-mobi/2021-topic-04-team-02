import numpy as np


def manhattan_distance(trainvalues_pca, x):
    """

    :param trainvalues_pca: array of training data
    :param x: tested data point
    :return: returns manhattan distance
    """
    distances = np.abs(trainvalues_pca - x).sum(-1)
    return distances


def euclidean_distance(trainvalues_pca, x):
    """

    :param trainvalues_pca: array of training data
    :param x: tested data point
    :return: returns euclidean distance
    """
    distances = np.linalg.norm(trainvalues_pca - x, axis=1)
    return distances


def voting_function(nearest_neighbours_labels):
    """

    :param nearest_neighbours_labels:
    :return:
    """
    counter_list_2 = []
    for value in np.unique(nearest_neighbours_labels):
        counter_list = []
        for i, number in enumerate(nearest_neighbours_labels):
            if number == value:
                counter_list.append(1)
        counter_list_2.append(np.sum(counter_list))
    index_maximum_value = np.argmax(counter_list_2)
    best_label = np.unique(nearest_neighbours_labels)[index_maximum_value]
    return best_label


def knn(distance_method_as_string, trainvalues_pca, trainlabels, x, k):
    """

    :param distance_method_as_string: euclidean or manhattan as a string
    :param trainvalues_pca: array of training data
    :param trainlabels: labels of the training data
    :param x: tested data point
    :param k: variable k for the knn algorithm
    :return: returns the major vote of the k nearest neighbours based on used distance method distance
    """
    if distance_method_as_string == "euclidean":
        distances = euclidean_distance(trainvalues_pca, x)
    elif distance_method_as_string == "manhattan":
        distances = manhattan_distance(trainvalues_pca, x)
    else:
        raise ValueError("Distance method not implemented, please use euclidean or manhattan!")
    nearest = trainlabels[np.argsort(distances)[:k]]

    return voting_function(nearest)


def weighted_knn(distance_method_as_string, trainvalues_pca, trainlabels, x, k):
    """

    :param distance_method_as_string: euclidean or manhattan as a string
    :param trainvalues_pca: array of training data
    :param trainlabels: labels of the training data
    :param x: tested data point
    :param k: variable k for the knn algorithm
    :return:
    """
    if distance_method_as_string == "euclidean":
        distances = euclidean_distance(trainvalues_pca, x)
    elif distance_method_as_string == "manhattan":
        distances = manhattan_distance(trainvalues_pca, x)
    else:
        raise ValueError("Distance method not implemented, please use euclidean or manhattan!")
    weights = 1.0 / distances**2
    weights /= weights.sum(axis=0)

    nearestweights = weights[np.argsort(distances)[:k]]
    nearest = trainlabels[np.argsort(distances)[:k]]

    weight_list_2 = []
    for value in np.unique(nearest):
        weight_list = []
        for i, number in enumerate(nearest):
            if number == value:
                weight_list.append(nearestweights[i])
        weight_list_2.append(np.sum(weight_list))
    index_maximum_value = np.argmax(weight_list_2)
    best_label = np.unique(nearest)[index_maximum_value]

    return best_label


def kdtree_knn(x, k, trainlabels, kdtree, distance_method):
    """

    :param x: tested data point
    :param k: variable k for the knn algorithm
    :param trainlabels: labels of the training data
    :param kdtree: pre calculated kd-tree
    :param distance_method:
    :return:
    """
    if distance_method == "euclidean":
        p = 2
    elif distance_method == "manhattan":
        p = 1
    else:
        raise ValueError("Distance method not implemented, please use euclidean or manhattan!")
    tree = kdtree
    dd, ii = tree.query(x, k=k, p=p)
    nearest = trainlabels[ii]

    return voting_function(nearest)


def distances_euclidean_testing(trainvalues_pca, trainlabels, testvalues_pca, k):
    """

    :param trainvalues_pca: array of training data
    :param trainlabels: labels of the training data
    :param testvalues_pca: array of testing data
    :param k: variable k for the knn algorithm
    :return:x should return labels of the nearest neighbours for all test data points at once
    """

    distances = np.sqrt(((trainvalues_pca[np.newaxis] - testvalues_pca[:, np.newaxis])**2).sum(axis=-1))
    nearest = trainlabels[np.argsort(distances)[:, :k]]
    return nearest
