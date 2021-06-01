import numpy as np
from numpy.linalg import norm

def knn(trainvalues, trainlabels, X,k):
    distances = norm(trainvalues-X, axis=1)
    nearest = trainlabels[np.argsort(distances)[:k]]
    return mode(nearest)[0][0]


