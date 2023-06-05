import numpy as np


def correlationMatrixReal(s, m_size):
    matrix = np.zeros((m_size, m_size))
    n_iter = s.shape[0] - m_size
    for i in range(n_iter):
        v = s[i : i + m_size]
        matrix += np.outer(v, v)

    return matrix

def solveEquation(corr_matrix):
    w = -np.linalg.pinv(corr_matrix[:-1, :-1]) @ corr_matrix[0, 1:]
    w = np.concatenate(([1.0], w))

    return w

def kForecast(s: np.ndarray, w: np.ndarray, k_steps: int):
    for _ in range(k_steps):
        s[-1] = np.sum(w * s)
        s = np.roll(s, 1)
    
    return s[0]