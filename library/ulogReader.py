"""
Module 
"""

import numpy as np
import quaternion as qt

from pyulog import ULog
from library.quaternionUtils import eulerToQuaternion



def readAttitude(ulog: ULog):
    """
    Function extract vehicle attitude from ulogs to quaternion.

    Args:
        ulg (pyulog.ULog): PyUlog object to read

    Returns:
        attitude (np.array(qt.quaternion)): Array of quaternions value (n,)
        timestamp (np.array): Array of timestamp (n,)
    """
    
    att_logs = ulog.get_dataset('vehicle_attitude') # Get vehicle_attitude dataset
    n_logs   = att_logs.data['q[0]'].shape[0] # N_logs
    att_quat  = np.zeros((n_logs,4)) # Allocate memory

    att_quat[:,0] = att_logs.data['q[0]']
    att_quat[:,1] = att_logs.data['q[1]']
    att_quat[:,2] = att_logs.data['q[2]']
    att_quat[:,3] = att_logs.data['q[3]']

    return qt.from_float_array(att_quat), att_logs.data['timestamp']


def readNavigator(ulg: ULog):
    """
    Read navigator signal from ulg file as euler angles in radians `(Roll, Pitch,  Yaw)`.
    Args:
        ulg (pyulog.ULog): PyUlog object to read.

    Returns:
        nav_euler (np.ndarray): Array of euler value (n, 3)
        time_stamp (np.array): Array of timestamp (n,)
    """   

    nav_logs = ulg.get_dataset('navigator_setpoint') # Get navigator_setpoint dataset
    n_logs   = nav_logs.data['sp_Roll_angle'].shape[0] # N_logs

    nav_euler  = np.zeros((n_logs,3)) # Allocate memory
    nav_euler[:,0] = nav_logs.data['sp_Roll_angle']
    nav_euler[:,1] = nav_logs.data['sp_Pitch_angle']
    nav_euler[:,2] = nav_logs.data['sp_Yaw_angle']

    return eulerToQuaternion(nav_euler, 'XYZ', False), nav_logs.data['timestamp']