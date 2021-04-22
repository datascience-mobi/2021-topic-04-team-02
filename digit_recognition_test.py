import numpy as np
import pickle
with open('data/train_points.p', 'rb') as f:
    X = pickle.load(f)
trainlabels = X[:,0]
trainvalues = X[:,1:]

with open('data/test_points.p', 'rb') as f:
    Y = pickle.load(f)
testvalues = Y[:,1:]
testlabels = Y[:,0]
testvalues.shape

from scipy.stats import mode
k=100
def predict(x):
    differences = (trainvalues - x)
    distances = np.einsum('ij, ij->i', differences, differences)
    nearest = trainlabels[np.argsort(distances)[:k]]
    return mode(nearest)[0][0]
hit = 0
miss = 0
for i in range(0,10000): #PCA definitely necessary
    sample = i
    predicted_value = predict(x=testvalues[sample,:])
    labeled_value = testlabels[sample]
    if predicted_value == labeled_value:
        hit +=1
    else:
        miss +=1
print(hit, 'vs',miss) #hit or miss, I guess they never miss huh
#992 vs 9007
#run time: 10 minutes at least
