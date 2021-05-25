import numpy as np
import pandas as pd

# code for standard scaler of the data
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
X_scaled[:5]

# calculating covariance matrix
features = X_scaled.T
cov_matrix = np.cov(features)
cov_matrix[:5]
kj