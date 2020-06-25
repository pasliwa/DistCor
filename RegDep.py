import numpy as np
from sklearn import linear_model
import DistCor
from functools import partial


def prepare_data_for_regressive_stat(x, y, z, data):
    X = np.vstack(np.vstack([data[i] for i in x])).T
    Y = np.vstack(np.vstack([data[i] for i in y])).T
    if z:
        Z = np.vstack(np.vstack([data[i] for i in z])).T
        return X, Y, Z
    else:
        return X, Y, None


def reg_correlation(X, Y, Z, model=linear_model.LassoLarsCV()):
    model.fit(Z, X.ravel())
    X_res = X.ravel() - model.predict(Z)

    model.fit(Z, Y.ravel())
    Y_res = Y.ravel() - model.predict(Z)
    return np.corrcoef(X_res.flatten(), Y_res.flatten())[0][1]


def partial_correlation(X, Y, Z):
    if Z:
        return reg_correlation(X, Y, Z, model=linear_model.LinearRegression())
    else:
        return np.corrcoef(X, Y)[0][1]


def test_stat_partial_correlation(x, y, z, data):
    X, Y, Z = prepare_data_for_regressive_stat(x, y, z, data)
    return partial_correlation(X, Y, Z)


def distance_resid_correlation(X, Y, Z, model):
    U_vector_A = DistCor.U_centered_matrix(DistCor.dist_matrix(X)).flatten('F').reshape(-1, 1)
    U_vector_B = DistCor.U_centered_matrix(DistCor.dist_matrix(Y)).flatten('F').reshape(-1, 1)
    if Z:
        flatten = partial(np.ndarray.flatten, order="F")
        U_matrix_C = np.vstack(list(map(flatten, map(DistCor.U_centered_matrix, map(DistCor.dist_matrix, Z.T))))).T
        return reg_correlation(U_vector_A, U_vector_B, U_matrix_C, model)
    else:
        return np.corrcoef(U_vector_A, U_vector_B)[0][1]


def test_stat_distance_resid_correlation(x, y, z, data, model=linear_model.LassoLarsCV()):
    X, Y, Z = prepare_data_for_regressive_stat(x, y, z, data)
    return distance_resid_correlation(X, Y, Z, model)


def background_null_distance_resid_correlation(x, y, z, data, model=linear_model.LassoLarsCV(), Reps=100):
    n = x.shape[0]
    backgrounds = []

    X, Y, Z = prepare_data_for_regressive_stat(x, y, z, data)
    UCM_A = DistCor.U_centered_matrix(DistCor.dist_matrix(X))
    U_vector_B = DistCor.U_centered_matrix(DistCor.dist_matrix(Y)).flatten('F').reshape(-1, 1)

    indices = np.arange(n)

    if Z:
        flatten = partial(np.ndarray.flatten, order="F")
        U_matrix_C = np.vstack(list(map(flatten, map(DistCor.U_centered_matrix, map(DistCor.dist_matrix, Z.T))))).T
        for i in range(Reps):
            np.random.shuffle(indices)
            shuffled_U_vector_A = UCM_A[:, indices][indices].flatten('F').reshape(-1, 1)
            background_statistic = reg_correlation(shuffled_U_vector_A, U_vector_B, U_matrix_C, model)
            backgrounds.append(background_statistic)
    else:
        for i in range(Reps):
            np.random.shuffle(indices)
            shuffled_U_vector_A = UCM_A[:, indices][indices].flatten('F').reshape(-1, 1)
            background_statistic = np.corrcoef(shuffled_U_vector_A, U_vector_B)[0][1]
            backgrounds.append(background_statistic)
    return backgrounds


def background_null_partial_correlation(x, y, z, data, Reps=100):
    n = x.shape[0]
    backgrounds = []

    X, Y, Z = prepare_data_for_regressive_stat(x, y, z, data)

    indices = np.arange(n)

    for i in range(Reps):
        np.random.shuffle(indices)
        shuffled_X = X[indices]
        background_statistic = partial_correlation(shuffled_X, Y, Z)
        backgrounds.append(background_statistic)
    return backgrounds
