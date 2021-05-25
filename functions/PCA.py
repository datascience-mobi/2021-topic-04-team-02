import numpy as np

#how to load dataset???
def I_turned_myself_into_a_pickle_morty(data,new_data_location):
    X = np.genfromtxt(data, delimiter=',').astype(np.dtype('uint8'))
    with open(new_data_location, 'wb') as f:
        pickle.dump(X, f)

# code for standard scaler of the data
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
X_scaled[:5]

# calculating covariance matrix
features = X_scaled.T
cov_matrix = np.cov(features)
cov_matrix[:5]

# Eigendecomposition
values, vectors = np.linalg.eig(cov_matrix)
values[:5]