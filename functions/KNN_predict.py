import numpy as np
from numpy.linalg import norm
from scipy.stats import mode

def knn(trainvalues, trainlabels, X, k):
    distances = norm(trainvalues-X, axis = 1) #calculates the euclidean distance between trainvalues and data point X
    nearest = trainlabels[np.argsort(distances)[:k]] #extracting the k nearest neighbours from the trainlabels through sorting the distances
    return mode(nearest)[0][0] #returning the major vote of the k nearest neighbours


