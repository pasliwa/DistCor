import numpy as np
from sklearn import linear_model
import DistCor
from functools import partial


def prepare_data_for_regression_stat(x, y, z, data):
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
    if np.isclose(np.linalg.norm(X_res), 0) or np.isclose(np.linalg.norm(Y_res), 0):
        return 0
    return np.corrcoef(X_res.flatten(), Y_res.flatten())[0][1]


def partial_correlation(X, Y, Z):
    if Z is not None:
        return reg_correlation(X, Y, Z, model=linear_model.LinearRegression())
    else:
        return np.corrcoef(X.flatten(), Y.flatten())[0][1]


def test_stat_partial_correlation(x, y, z, data):
    X, Y, Z = prepare_data_for_regression_stat(x, y, z, data)
    return partial_correlation(X, Y, Z)


def distance_resid_correlation(X, Y, Z, model):
    UCMD_A = DistCor.U_centered_matrix(DistCor.dist_matrix(X))
    UCMD_B = DistCor.U_centered_matrix(DistCor.dist_matrix(Y))
    triu_indices = np.triu_indices_from(UCMD_A, k=1)
    U_vector_A = UCMD_A[triu_indices].reshape(-1, 1)
    U_vector_B = UCMD_B[triu_indices].reshape(-1, 1)
    if Z is not None:
        UCMD_Cs = list(map(DistCor.U_centered_matrix, map(DistCor.dist_matrix, Z.T)))
        U_matrix_C = np.vstack([UCMD_Cs[i][triu_indices] for i in range(len(UCMD_Cs))]).T
        return reg_correlation(U_vector_A, U_vector_B, U_matrix_C, model)
    else:
        return np.corrcoef(U_vector_A.flatten(), U_vector_B.flatten())[0][1]


def test_stat_distance_resid_correlation(x, y, z, data, model=linear_model.LassoLarsCV()):
    X, Y, Z = prepare_data_for_regression_stat(x, y, z, data)
    return distance_resid_correlation(X, Y, Z, model)


def background_null_distance_resid_correlation(x, y, z, data, model=linear_model.LassoLarsCV(), Reps=100):
    backgrounds = []

    X, Y, Z = prepare_data_for_regression_stat(x, y, z, data)
    n = X.shape[0]
    UCMD_A = DistCor.U_centered_matrix(DistCor.dist_matrix(X))
    triu_indices = np.triu_indices_from(UCMD_A, k=1)
    UCMD_B = DistCor.U_centered_matrix(DistCor.dist_matrix(Y))
    U_vector_B = UCMD_B[triu_indices].reshape(-1, 1)

    indices = np.arange(n)

    if Z is not None:
        UCMD_Cs = list(map(DistCor.U_centered_matrix, map(DistCor.dist_matrix, Z.T)))
        U_matrix_C = np.vstack([UCMD_Cs[i][triu_indices] for i in range(len(UCMD_Cs))]).T
        for i in range(Reps):
            np.random.shuffle(indices)
            shuffled_UCMD_A = UCMD_A[:, indices][indices]
            shuffled_U_vector_A = shuffled_UCMD_A[triu_indices].reshape(-1, 1)

            background_statistic = reg_correlation(shuffled_U_vector_A, U_vector_B, U_matrix_C, model)
            backgrounds.append(background_statistic)
    else:
        for i in range(Reps):
            np.random.shuffle(indices)
            shuffled_UCMD_A = UCMD_A[:, indices][indices]
            shuffled_U_vector_A = shuffled_UCMD_A[triu_indices].reshape(-1, 1)
            background_statistic = np.corrcoef(shuffled_U_vector_A.flatten(), U_vector_B.flatten())[0][1]
            backgrounds.append(background_statistic)
    return backgrounds


def background_null_partial_correlation(x, y, z, data, Reps=100):
    backgrounds = []

    X, Y, Z = prepare_data_for_regression_stat(x, y, z, data)
    n = X.shape[0]
    indices = np.arange(n)

    for i in range(Reps):
        np.random.shuffle(indices)
        shuffled_X = X[indices]
        background_statistic = partial_correlation(shuffled_X, Y, Z)
        backgrounds.append(background_statistic)
    return backgrounds

def test_distance_resid_correlation_cond_Z(x, y, z, data, model = linear_model.LassoLarsCV(), Reps=100):
    return test_stat_distance_resid_correlation(x, y, z, data, model), \
           background_null_distance_resid_correlation(x, y, z, data, model, Reps)

def test_partial_correlation_cond_Z(x, y, z, data, Reps=100):
    return test_stat_partial_correlation(x, y, z, data), \
           background_null_partial_correlation(x, y, z, data, Reps)
