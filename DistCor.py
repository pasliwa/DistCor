import numpy as np


def dist_matrix(X):
    """
    Given an array (X) of points, calculates the pairwise distances
    """
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    s = np.sum(X ** 2, axis=1)
    squared_distance = s + s[:, np.newaxis] - 2 * np.dot(X, X.T)
    squared_distance[np.isclose(squared_distance, 0)] = 0
    return np.sqrt(squared_distance)


def U_centered_matrix(a):
    """
    Given a distance matrix, return its U_centered_matrix as in Szekely and Rizzo
    """
    n = a.shape[0]
    sumcol = a.sum(axis=0)
    grandsum = sumcol.sum()
    U_cm = a - (sumcol / (n - 2)) - (sumcol[:, np.newaxis] / (n - 2)) + (grandsum / ((n - 1) * (n - 2)))
    np.fill_diagonal(U_cm, 0)
    return U_cm


def D_centered_matrix(a):
    """
    Given a distance matrix, return its D_centered_matrix
    """
    n = a.shape[0]
    sumcol = a.sum(axis=0)
    grandsum = sumcol.sum()
    return a - (sumcol / n) - (sumcol[:, np.newaxis] / n) + (grandsum / (n ** 2))


def inner_product(A, B):
    n = np.shape(A)[0]
    return np.sum(A * B) / (n * (n - 3))


def project(A, C, threshold=0.000001):
    if inner_product(C, C) <= threshold:
        return A
    else:
        return A - (inner_product(A, C) / inner_product(C, C)) * C


def squared_pop_dCov(x, y):
    UCM_A = U_centered_matrix(dist_matrix(x))
    UCM_B = U_centered_matrix(dist_matrix(y))
    return inner_product(UCM_A, UCM_B)


def pdCov(x, y, z):
    UCM_A = U_centered_matrix(dist_matrix(x))
    UCM_B = U_centered_matrix(dist_matrix(y))
    UCM_C = U_centered_matrix(dist_matrix(z))

    Pz_x = project(UCM_A, UCM_C)
    Pz_y = project(UCM_B, UCM_C)
    return inner_product(Pz_x, Pz_y)


def dcov_test_statistic(x, y):
    n = x.shape[0]
    UCM_A = U_centered_matrix(dist_matrix(x))
    UCM_B = U_centered_matrix(dist_matrix(y))

    test_stat = n * inner_product(UCM_A, UCM_B)
    return test_stat, UCM_A, UCM_B


def pdcov_test_statistic(x, y, z):
    n = x.shape[0]

    UCM_A = U_centered_matrix(dist_matrix(x))
    UCM_B = U_centered_matrix(dist_matrix(y))
    UCM_C = U_centered_matrix(dist_matrix(z))

    Pz_x = project(UCM_A, UCM_C)
    Pz_y = project(UCM_B, UCM_C)
    test_stat = n * inner_product(Pz_x, Pz_y)  # n * pdCov(x, y, z)
    return test_stat, Pz_x, Pz_y


def squared_pop_dCov_with_background(x, y, Reps=100):
    n = x.shape[0]
    test_stat, UCM_A, UCM_B = dcov_test_statistic(x, y)

    backgrounds = []
    indices = np.arange(n)

    for i in range(Reps):
        np.random.shuffle(indices)
        background_statistic = n * inner_product(UCM_A[:, indices][indices], UCM_B)
        backgrounds.append(background_statistic)
    return test_stat, backgrounds


def pdCov_with_background(x, y, z, Reps=100):
    n = x.shape[0]
    test_stat, Pz_x, Pz_y = pdcov_test_statistic(x, y, z)

    backgrounds = []
    indices = np.arange(n)

    for i in range(Reps):
        np.random.shuffle(indices)
        background_statistic = n * inner_product(Pz_x[:, indices][indices], Pz_y)
        backgrounds.append(background_statistic)
    return test_stat, backgrounds


def test_partial_dCov_cond_Z(x, y, z, data, Reps=500):
    X = np.vstack(np.vstack([data[i] for i in x])).T
    Y = np.vstack(np.vstack([data[i] for i in y])).T
    if z:
        Z = np.vstack(np.vstack([data[i] for i in z])).T
        return pdCov_with_background(X, Y, Z, Reps)
    else:
        return squared_pop_dCov_with_background(X, Y, Reps)


def test_stat_squared_pop_dCov(x, y):
    n = x.shape[0]
    UCM_A = U_centered_matrix(dist_matrix(x))
    UCM_B = U_centered_matrix(dist_matrix(y))

    test_stat = n * inner_product(UCM_A, UCM_B)  # n * pdCov(x, y, z)
    return test_stat


def test_stat_pdCov(x, y, z):
    n = x.shape[0]

    UCM_A = U_centered_matrix(dist_matrix(x))
    UCM_B = U_centered_matrix(dist_matrix(y))
    UCM_C = U_centered_matrix(dist_matrix(z))

    Pz_x = project(UCM_A, UCM_C)
    Pz_y = project(UCM_B, UCM_C)
    test_stat = n * inner_product(Pz_x, Pz_y)  # n * pdCov(x, y, z)
    return test_stat


def give_test_statistic(x, y, z, data):
    X = np.vstack(np.vstack([data[i] for i in x])).T
    Y = np.vstack(np.vstack([data[i] for i in y])).T
    if z:
        Z = np.vstack(np.vstack([data[i] for i in z])).T
        return test_stat_pdCov(X, Y, Z)
    else:
        return test_stat_squared_pop_dCov(X, Y)
