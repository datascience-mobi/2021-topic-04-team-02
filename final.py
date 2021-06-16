import numpy as np
import pickle
from functions.Load_data import load_the_pickle
from functions.PCA import PCA_func
from functions.KNN_predict import *
from functions.Standardize import center
from functions.Standardize import center_test_values

# select number of principle components and k:
number_of_pcs = 45
k = 6

# loading data:
train_labels, train_values = load_the_pickle('data/train_points.p')
test_labels, test_values = load_the_pickle('data/test_points.p')

# standardization and PCA:
train_values_centered, train_mean = center(train_values)
train_values_pca, train_evs = PCA_func(train_values_centered, train_mean, number_of_pcs)

test_values_centered, test_mean = center_test_values(test_values, train_values)
test_values_pca, _ = PCA_func(test_values_centered,test_mean, number_of_pcs, train_evs=train_evs)

# kNN:
hit = 0
miss = 0

for sample in range(10000):
    predicted_value = knn_euclidean(trainvalues_pca=train_values_pca, X=test_values_pca[sample,:], trainlabels=train_labels, k=k)
    labeled_value = test_labels[sample]
    if predicted_value == labeled_value:
        hit += 1
    else:
        miss += 1
print(hit, 'vs', miss)

# Results:
# 9739 vs 261 ->k=6 pc=45
# run time: 6:10 min
