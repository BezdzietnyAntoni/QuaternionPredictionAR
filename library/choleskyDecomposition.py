"""
Module implements Cholesky decomposition schema for quaternion 
symmetric hermitian matrix and function to solve matrix equation.
"""

import numpy as np
import quaternion as qt
from scipy.linalg import fractional_matrix_power


def choleskyDecomposition(a: np.ndarray):
    """
    Cholesky factorization for quaternion symmetric hermitian matrix.
    Base on "A structure-preserving algorithm for the quaternion Cholesky decomposition"
        https://www.sciencedirect.com/science/article/abs/pii/S009630031300894

    Args:
        a (np.ndarray(qt.quaternion)): Symmetric 2 dimensional quaternion matrix

    Returns:
        l (np.ndarray(qt.quaternion)): Lower triangular matrix
    """

    if a.ndim != 2:
        raise ValueError('Array a must have 2 dimensional')
    

    m = qt.as_float_array(a)

    a1 = m[:,:,0]
    a2 = m[:,:,1]
    a3 = m[:,:,2]
    a4 = m[:,:,3]

    n = m.shape[0]

    for j in range(n-1):
        for i in range(j+1, n):
            l1 = -a1[i,j] / a1[j,j]
            l2 = -a2[i,j] / a1[j,j]
            l3 = -a3[i,j] / a1[j,j]
            l4 = -a4[i,j] / a1[j,j]

            a1[i,i] = a1[i,i] + a1[j,j] * (l1*l1 + l2*l2 + l3*l3 + l4*l4) + \
                2 * (l1*a1[i,j] + l2*a2[i,j] + l3*a3[i,j] + l4*a4[i,j])

            for k in range(i+1, n):
                a1[k,i] = a1[k,i] + l1*a1[k,j] + l2*a2[k,j] + l3*a3[k,j] + l4*a4[k,j]
                a2[k,i] = a2[k,i] - l2*a1[k,j] + l1*a2[k,j] - l4*a3[k,j] + l3*a4[k,j]
                a3[k,i] = a3[k,i] - l3*a1[k,j] + l4*a2[k,j] + l1*a3[k,j] - l2*a4[k,j]
                a4[k,i] = a4[k,i] - l4*a1[k,j] - l3*a2[k,j] + l2*a3[k,j] + l1*a4[k,j]
            
            a1[i,j] = -l1
            a2[i,j] = -l2
            a3[i,j] = -l3
            a4[i,j] = -l4

    l1 = np.tril(a1, -1) + np.eye(n)
    l2 = np.tril(a2, -1)
    l3 = np.tril(a3, -1)
    l4 = np.tril(a4, -1)

    d = fractional_matrix_power((np.diag(np.diag(a1))), 0.5)

    l = np.zeros((*l1.shape, 4))

    l[:,:,0] = l1 @ d
    l[:,:,1] = l2 @ d
    l[:,:,2] = l3 @ d
    l[:,:,3] = l4 @ d

    return qt.from_float_array(l)



def forwardSubstitution(l_tri: np.ndarray, b: np.ndarray):
    """
    Left forward substitution for lower triangular matrix.
    Ly = b <- equation solved 

    Args:
        l_tri (np.ndarray(qt.quaternion)): Lower triangular matrix (n,n)
        b (np.ndarray(qt.quaternion)): Equation values (n,)
    
    Returns:
        y (np.ndarray(qt.quaternion)): Solve values (n,)
    """
    y = np.zeros(b.shape[0], dtype=qt.quaternion)

    for i in range(y.shape[0]):
        y[i] = l_tri[i,i].inverse() * (b[i] - np.sum(l_tri[i,:i] * y[:i]))

    return y   


def backwardSubstitution(u_tri: np.ndarray, y: np.ndarray):
    """
    Left backward substitution for upper triangular matrix.
    Ux = y <- equation solved 

    Args:
        u_tri (np.ndarray(qt.quaternion)): Upper triangular matrix (n,n)
        y (np.ndarray(qt.quaternion)): Equation values (n,)
    
    Returns:
        x (np.ndarray(qt.quaternion)): Solve values (n,)
    """   
    x = np.zeros(y.shape, dtype=qt.quaternion)

    last_idx = y.shape[0] - 1
    x[last_idx] = u_tri[last_idx, last_idx].inverse() * y[last_idx] 

    for i in range(last_idx-1, -1, -1):
        x[i] = u_tri[i, i].inverse() * (y[i] - np.sum(u_tri[i, i+1:] * x[i+1:])) 

    return x


def solveLLHEquation(l_tri: np.ndarray, b:np.ndarray):
    """
    Solve equation in formula L(L^H)x = b in left order schema.
    
    Args:
        l_tri (np.ndarray(qt.quaternion)): Lower triangular matrix (n,n)
        b (np.ndarray(qt.quaternion)): Equation values (n,)

    Returns:
        x (np.ndarray(qt.quaternion)): Solve values (n,)
    """

    y = forwardSubstitution(l_tri, b)
    x = backwardSubstitution(l_tri.conjugate().T, y)
    return x   
