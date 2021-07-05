import numpy as np
import pickle
from functions.Load_data import load_the_pickle
from functions.PCA import PCA_func
from functions.KNN_predict import *
from functions.Standardize import center

# select number of principle components and k:
number_of_pcs = 45
k = 6

# loading data:
train_labels, train_values = load_the_pickle('data/train_points.p')
test_labels, test_values = load_the_pickle('data/test_points.p')

# standardization and PCA:
train_values_centered, train_mean = center(train_values, Y="None")
train_values_pca, train_evs = PCA_func(train_values_centered, train_mean, number_of_pcs)

test_values_centered, test_mean = center(test_values, Y=train_values)
test_values_pca, _ = PCA_func(test_values_centered,test_mean, number_of_pcs, train_evs=train_evs)

def knn_multi(x):
    return knn(distance_method_as_string="euclidean",trainvalues_pca=train_values_pca, X=test_values_pca[x,:], trainlabels=train_labels, k=k)
# kNN:
hit = 0
miss = 0

import multiprocessing

l = list(range(10000))
if __name__ == '__main__':
    with multiprocessing.Pool(10) as p:
        result = p.map(knn_multi, range(10000))
    for sample in range(10000):
        if result[sample] == test_labels[sample]:
            hit += 1
        else:
            miss +=1
    print(hit, "vs", miss)

# Results:
# 9739 vs 261 --> k=6 pc=45 --> euclidean
# run time: 113 Sekunden -->4 Prozesse
# 9713 vs 287 --> k=6 pc=45 --> manhattan
# run time: 110 Sekunden -->4 Prozesse