import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode

def I_turned_myself_into_a_pickle_morty(data,new_data_location):
    X = np.genfromtxt(data, delimiter=',').astype(np.dtype('uint8'))
    with open(new_data_location, 'wb') as f:
        pickle.dump(X, f)

def do_pca(n_components, data): #does pca + data standardization I think?
    X = StandardScaler().fit_transform(data) #
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca
def load_the_training_pickle(pickleddata):
    with open(pickleddata, 'rb') as f:
        X = pickle.load(f)
    global trainlabels
    trainlabels = X[:,0]
    global trainvalues
    trainvalues = X[:,1:]
def load_the_testing_pickle(pickleddata):
    with open(pickleddata, 'rb') as f:
        Y = pickle.load(f)
    global testvalues
    testvalues = Y[:,1:]
    global testlabels
    testlabels = Y[:,0]
def predict(x):
    differences = (trainvalues_pca - x)
    distances = np.einsum('ij, ij->i', differences, differences) #
    nearest = trainlabels[np.argsort(distances)[:k]]
    return mode(nearest)[0][0]

#K + Anzahl der Hauptkomponenten festlegen:
number_of_pcs = 8
k=150
hit = 0
miss = 0

load_the_training_pickle('data/train_points.p')
load_the_testing_pickle('data/test_points.p')

pca, testvalues_pca = do_pca(number_of_pcs,testvalues)
pca, trainvalues_pca = do_pca(number_of_pcs,trainvalues)



for i in range(0,10000): #PCA helps a lot
    sample = i
    predicted_value = predict(x=testvalues_pca[sample,:])
    labeled_value = testlabels[sample]
    if predicted_value == labeled_value:
        hit +=1
    else:
        miss +=1
print(hit, 'vs',miss) #hit or miss, I guess they never miss huh

#7865 vs 2135 ->k=150 pc=8
#run time: 5 min
