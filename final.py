import numpy as np
import pickle
from functions.Load_data import load_the_pickle
from functions.PCA import PCA_func
import functions.KNN_predict as knn
from functions.Standardize import center
import itertools as itertools
import multiprocessing
from scipy.spatial import KDTree
from scipy.stats import mode
import random

# select number of principle components and k:
number_of_pcs = 45
k = 6

# loading data:
train_labels, train_values = load_the_pickle('data/train_points.p')
test_labels, test_values = load_the_pickle('data/test_points.p')

# standardization and PCA:
train_values_centered, train_mean = center(train_values)
train_values_pca, train_evs = PCA_func(train_values_centered, train_mean, number_of_pcs)

test_values_centered, test_mean = center(test_values, Y=train_values)
test_values_pca, _ = PCA_func(test_values_centered,test_mean, number_of_pcs, train_evs=train_evs)

# kNN:
hit = 0
miss = 0

#tree = KDTree(train_values_pca)
#if __name__ == '__main__':
#    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
#        result = p.starmap(knn.kdtree_knn,zip(test_values_pca[range(10000),:],itertools.repeat(k),itertools.repeat(train_labels),itertools.repeat(tree)),chunksize=500)
#        for sample in range(10000):
#            if result[sample] == test_labels[sample]:
#                hit += 1
#            else:
#                miss +=1
#        print(hit, "vs", miss)

if __name__ == '__main__':
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        result = p.starmap(knn.weighted_knn, zip(itertools.repeat("euclidean"), itertools.repeat(train_values_pca),itertools.repeat(train_labels),test_values_pca[range(10000),:],itertools.repeat(k)),chunksize=500)

        for sample in range(10000):
            if result[sample] == test_labels[sample]:
                hit += 1
            else:
                miss +=1
        print(hit, "vs", miss)


# Results:
# 9739 vs 261 --> k=6 pc=45 --> euclidean
# run time: 99 Sekunden -->4 Prozesse
# 9713 vs 287 --> k=6 pc=45 --> manhattan
# run time: 97 Sekunden -->4 Prozesse

