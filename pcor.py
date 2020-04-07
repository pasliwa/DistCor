import numpy as np
import pandas as pd
from sklearn import linear_model


def partial_correlation(X, Y, Z):
    model = linear_model.LinearRegression()
    model.fit(Z, X)
    X_res = X - model.predict(Z)
    model = linear_model.LinearRegression()
    model.fit(Z, Y)
    Y_res = Y - model.predict(Z)
    return np.corrcoef(X_res.flatten(), Y_res.flatten())

def give_test_statistic_pc(x, y, z, data):
    X = np.vstack(np.vstack([data[i] for i in x])).T
    Y = np.vstack(np.vstack([data[i] for i in y])).T
    if z:
        Z = np.vstack(np.vstack([data[i] for i in z])).T
        return partial_correlation(X, Y, Z)
    else:
        return np.corrcoef(X.flatten(), Y.flatten())


