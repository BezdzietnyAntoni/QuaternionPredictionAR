"""
Module implements functions used for quaternion processed.
"""

import numpy as np
import quaternion as qt

from scipy.spatial.transform import Rotation



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