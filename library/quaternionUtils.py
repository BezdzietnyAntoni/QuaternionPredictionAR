"""
Module implements functions used for quaternion processed.
"""

import random
import numpy as np
import quaternion as qt

from scipy.spatial.transform import Rotation
from quaternion.calculus import spline_evaluation


def quaternionToEuler(quaternion: np.ndarray | qt.quaternion, 
                      seq: str = 'XYZ',
                      degrees: bool = True):
    """
    Convert quaternion to Euler angles. 
    Sequence `seq` defines Euler angles order.

    Args:
        quaternion (np.ndarray(qt.quaternion) | qt.quaternion): Quaternion 
        seq (str, optional): Euler angles sequence. Defaults to 'XYZ'.
        degrees (bool, optional): If True return value in degrees. Defaults to True.

    Returns:
        euler_angle (np.ndarray): Euler angles (3,n)
    """
    
    # If it's a single qt.quaternion or list of qt.quaternions
    if isinstance(quaternion, (qt.quaternion, list)): 
        quaternion = np.array(quaternion)

    # Cast qt.quaternion(n,) -> float(4,n)
    quaternion = qt.as_float_array(quaternion)

    # Extension for min 2 dimensional
    if quaternion.ndim == 1:
        quaternion = np.array(quaternion)[np.newaxis]

    # Transform to Euler angles
    quaternion = np.roll(quaternion, -1, axis=1) # Cast to formula (x, y, z, w)
    return Rotation.from_quat(quaternion).as_euler(seq, degrees)


def eulerToQuaternion(euler: np.ndarray,  seq: str = 'XYZ', degrees: bool = True):
    """
    Function cast euler angles given by array (3,) or (n,3) in sequence given by `seq` parameters 
    to quaternion.

    Args:
        euler (list | np.array): Euler angles array (3,) or (n,3) 
        seq (str, optional): Euler sequence. Defaults to 'XYZ'.
        degrees (bool, optional): Set true if given value `euler` are in degree. Defaults to True.

    Returns:
        quaternion: Quaternions array (n,)
    """

    # If it's a list
    if isinstance(euler, list):
        euler = np.array(euler)

    # Extension for min 2 dimensional
    if euler.ndim == 1:
        euler = np.array(euler)[np.newaxis]

    # Transform to quaternion
    quaternion = Rotation.from_euler(seq, euler, degrees).as_quat()
    quaternion = np.roll(quaternion, 1, axis=1)# Cast to formula (w, x, y, z)
    return  qt.from_float_array(quaternion) 


def quaternionMatrixMultiply(q1: np.ndarray, q2: np.ndarray):
    """
    Multiply quaternion matrix 

    Args:
        q1 (np.ndarray): Matrix of quaternions (n,m)
        q2 (np.ndarray): Matrix of quaternions (m,k)

    Returns:
        matrix: Matrix of quaternions (n,k)
    """
    if q1.shape[1] != q2.shape[0]:
        raise 'q1.shape[1] must be equal q2.shape[0]'
    
    matrix = np.zeros((q1.shape[0],q2.shape[1]), dtype=qt.quaternion)
    
    for j in range(matrix.shape[0]):
        for i in range(matrix.shape[1]):
            matrix[j,i] = np.sum(q1[j,:] * q2[:,i])

    return matrix   


def splineInterpolation(quaternions: np.ndarray,
                        timestamp: np.ndarray,
                        interpolation_timestamp: np.ndarray,
                        spline_degree: int = 1):
    
    if timestamp[0] > interpolation_timestamp[0]:
        raise ValueError("'interpolation_timestamp' minimal value\
                          must be equal or greater than minimal value 'timestamp'")
    if timestamp[-1] < interpolation_timestamp[-1]:
        raise ValueError("'interpolation_timestamp' maximum value\
                          must be equal or less than maximum value 'timestamp'")

    # Normalization timestamp [0...1]
    denominator = timestamp[-1] - timestamp[0]

    timestamp_normalized = ( timestamp - timestamp[-1] ) / denominator
    interpolation_timestamp_normalized  = ( interpolation_timestamp - timestamp[-1] ) / denominator

    # Spline interpolation
    return spline_evaluation(quaternions, timestamp_normalized, 
                             interpolation_timestamp_normalized, 
                             spline_degree = spline_degree)


def correlationMatrix(s : np.ndarray, m_size : int, normalize : bool = False):
    """
    Designate correlation matrix for quaternion signal. 
    Base on  "Convolution and Correlation Based on Discrete Quaternion Fourier Transform"
        https://core.ac.uk/download/pdf/25495083.pdf

    Args:
        s (np.ndarray): Windowed signal
        m_size (int): Correlation matrix size
        normalize (bool, optional): Normalization flag, if `True` normalization on. Defaults to False.

    Returns:
        corr_matrix(np.ndarray): Correlation matrix (m,m)
    """
    corr_matrix = np.zeros((m_size, m_size), dtype = qt.quaternion)
    n_iter = s.shape[0] - m_size
    for i in range(n_iter):
        v = s[i : i + m_size]
        corr_matrix += np.outer(v, v.conjugate())
    
    if normalize:
        return corr_matrix / n_iter
    
    return corr_matrix


def randQuaternion():
    """
    Generate random normalized quaternion

    Returns:
        np.quaternion: Normalized quaternion
    """
    return np.quaternion(random.random(),
                         random.random(),
                         random.random(),
                         random.random()).normalized()


def randQuaternionVector(n: int):
    """
    Generate random vector (n,) normalized quaternion

    Args:
        n (int): Vector length

    Returns:
        np.array(qt.quaternion): Vector (n,) of normalized quaternion 
    """
    return np.array([randQuaternion() for _ in range(n)])
