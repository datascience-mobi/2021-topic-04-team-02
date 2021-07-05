import numpy as np
import pickle
from functions.Load_data import load_the_pickle
from functions.PCA import PCA_func
from functions.KNN_predict import *
from functions.Standardize import center


def demo(n, scale):
    # old version of final, placeholder until multiprocessing is cleaned up

    # select number of principle components and k:
    number_of_pcs = 45
    k = 6

    # loading data:
    train_labels, train_values = load_the_pickle('data/train_points.p')
    test_labels, test_values = load_the_pickle('data/test_points.p')

    # standardization and PCA:
    train_values_centered, train_mean = center(train_values, Y="None", scale=scale)
    train_values_pca, train_evs = PCA_func(train_values_centered, train_mean, number_of_pcs)

    test_values_centered, test_mean = center(test_values, Y=train_values, scale=scale)
    test_values_pca, _ = PCA_func(test_values_centered, test_mean, number_of_pcs, train_evs=train_evs)

    # kNN:
    hit = 0
    miss = 0
    miss_number = np.array([])

    for sample in range(n):
        predicted_value = knn("euclidean", trainvalues_pca=train_values_pca, X=test_values_pca[sample, :],
                              trainlabels=train_labels, k=k)
        labeled_value = test_labels[sample]
        if predicted_value == labeled_value:
            hit += 1
        else:
            miss += 1
            miss_number = np.append(miss_number, labeled_value)

    return miss_number, hit, miss
