import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def do_pca(n_components, data): #does pca + data standardization I think?
    X = StandardScaler().fit_transform(data)
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

with open('data/train_points.p', 'rb') as f:
    X = pickle.load(f)
trainlabels = X[:,0]
trainvalues = X[:,1:]

pca, trainvalues_pca = do_pca(59,trainvalues)

with open('data/test_points.p', 'rb') as f:
    Y = pickle.load(f)
testvalues = Y[:,1:]
testlabels = Y[:,0]
testvalues.shape

pca, testvalues_pca = do_pca(59,testvalues)

from scipy.stats import mode
k=100
def predict(x):
    differences = (trainvalues_pca - x)
    distances = np.einsum('ij, ij->i', differences, differences)
    nearest = trainlabels[np.argsort(distances)[:k]]
    return mode(nearest)[0][0]
hit = 0
miss = 0
for i in range(0,9999): #PCA helps a lot
    sample = i
    predicted_value = predict(x=testvalues_pca[sample,:])
    labeled_value = testlabels[sample]
    if predicted_value == labeled_value:
        hit +=1
    else:
        miss +=1
print(hit, 'vs',miss) #hit or miss, I guess they never miss huh

#5390 vs 4609
#run time: 5 minutes at least
