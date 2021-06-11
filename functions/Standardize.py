import numpy as np
from sklearn.preprocessing import MinMaxScaler


def stan(X):
    # centers and scales intensity values by column/ pixel position
    if np.std(X, axis=0) != 0:
        x_stan = ((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    else:
        x_stan = X - np.mean(X, axis=0)
    # If np.std of a column is 0 then only rescale
    return x_stan


def scale(X, min, max):
    scaler = MinMaxScaler(feature_range=(min, max))
    X_scale = scaler.transform(X)
    # documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    return X_scale

