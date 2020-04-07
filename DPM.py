import numpy as np
import pandas as pd
from sklearn import linear_model
import DistCor
from functools import partial


def partial_correlation(X, Y, Z, model=linear_model.LassoLarsCV()):
    #model = linear_model.LinearRegression()
    model.fit(Z, X.ravel())
    X_res = X.ravel() - model.predict(Z)
    #model = linear_model.LinearRegression()
    model.fit(Z, Y.ravel())
    Y_res = Y.ravel() - model.predict(Z)
    return np.corrcoef(X_res.flatten(), Y_res.flatten())


def give_test_statistic_DPM(x, y, z, data):
    X = np.vstack(np.vstack([data[i] for i in x])).T
    Y = np.vstack(np.vstack([data[i] for i in y])).T
    if z:
        Z = np.vstack(np.vstack([data[i] for i in z])).T
        return give_test_stat_DPM(X, Y, Z)
    else:
        return np.corrcoef(X.flatten(), Y.flatten())

def give_test_stat_DPM(x, y, z):
    n = x.shape[0]

    U_vector_A = DistCor.U_centered_matrix(DistCor.dist_matrix(x)).flatten('F').reshape(-1, 1)
    U_vector_B = DistCor.U_centered_matrix(DistCor.dist_matrix(y)).flatten('F').reshape(-1, 1)
    flatten = partial(np.ndarray.flatten, order="F")
    U_matrix_C = np.vstack(list(map(flatten, map(DistCor.U_centered_matrix, map(DistCor.dist_matrix, z.T))))).T

    return partial_correlation(U_vector_A, U_vector_B, U_matrix_C), U_vector_A, U_vector_B, U_matrix_C

def pval_test_stat_DPM(x, y, z, Reps=100):
    n = x.shape[0]

    test_stat, U_vector_A, U_vector_B, U_matrix_C = give_test_stat_DPM(x, y, z)

    backgrounds = []
    indices = np.arange(n)

    for i in range(Reps):
        np.random.shuffle(indices)
        shuffled_U_vector_A = U_vector_A.reshape(n, n).T[:, indices][indices].flatten('F').reshape(-1, 1)
        background_statistic = partial_correlation(shuffled_U_vector_A, U_vector_B, U_matrix_C)
        backgrounds.append(background_statistic)
    return test_stat, backgrounds



def give_pval_DPM(x, y, z, data):
    X = np.vstack(np.vstack([data[i] for i in x])).T
    Y = np.vstack(np.vstack([data[i] for i in y])).T
    if z:
        Z = np.vstack(np.vstack([data[i] for i in z])).T
        return pval_test_stat_DPM(X, Y, Z)
    else:
        return #pval_cor(X.flatten(), Y.flatten())

"""

def give_test_statistic_dpm(x, y, z, data):
    X = np.vstack(np.vstack([data[i] for i in x])).T
    Y = np.vstack(np.vstack([data[i] for i in y])).T
    if z:
        Z = np.vstack(np.vstack([data[i] for i in z])).T
        return give_test_stat_DPM(X, Y, Z)
    else:
        return np.corrcoef(X.flatten(), Y.flatten())

def give_test_stat_DPM(x, y, z):
    n = x.shape[0]

    UCM_A = DistCor.U_centered_matrix(DistCor.dist_matrix(x))
    UCM_B = DistCor.U_centered_matrix(DistCor.dist_matrix(y))
    UCM_C = DistCor.U_centered_matrix(DistCor.dist_matrix(z))
    return partial_correlation(UCM_A.flatten('F').reshape(-1,1), UCM_B.flatten('F').reshape(-1,1), UCM_C.flatten('F').reshape(-1,1))
"""