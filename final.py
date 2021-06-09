import numpy as np
import pickle
from functions.PCA import PCA_func
from functions.KNN_predict import knn

def I_turned_myself_into_a_pickle_morty(data,new_data_location):
    X = np.genfromtxt(data, delimiter=',').astype(np.dtype('uint8'))
    with open(new_data_location, 'wb') as f:
        pickle.dump(X, f)


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


#K und Anzahl der Hauptkomponenten festlegen:
number_of_pcs = 45
k = 6
hit = 0
miss = 0

load_the_training_pickle('data/train_points.p')
load_the_testing_pickle('data/test_points.p')

trainvalues_pca, train_mean, train_evs = PCA_func(trainvalues, number_of_pcs)
testvalues_pca, _, _ = PCA_func(testvalues, number_of_pcs, train_mean=train_mean, train_evs = train_evs)

for i in range(10000): #PCA helps a lot
    sample = i
    predicted_value = knn(trainvalues_pca = trainvalues_pca, X=testvalues_pca[sample,:], trainlabels = trainlabels, k = k)
    labeled_value = testlabels[sample]
    if predicted_value == labeled_value:
        hit +=1
    else:
        miss +=1
print(hit, 'vs',miss) #hit or miss, I guess they never miss huh

#9739 vs 261 ->k=6 pc=45
#run time: 5 min
