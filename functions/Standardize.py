import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


def stan(X):
    # centers and scales intensity values by column/ pixel position
    X_stan = ((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    # If np.std of a column is 0 then only rescale! How to implement?
    return X_stan


def scale(X, min, max):
    scaler = MinMaxScaler(feature_range=(min, max))
    X_scale = scaler.transform(X)
    # documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    return X_scale

def robust(X):
    rscaler = RobustScaler()
    X_rscale = rscaler.transform(X)
    # doc: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    return X_rscale