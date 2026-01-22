#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


#%% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    R = Rotation.from_quat(q.as_quat()).as_matrix()

    p  = p + v*dt + (0.5*(R@(a_m -a_b)+ g))*dt**2
    v  = v + (R@(a_m - a_b) + g)*dt
    q =  Rotation.from_matrix(R @  Rotation.from_rotvec(((w_m - w_b)*dt).flatten()).as_matrix())
    a_b = a_b
    w_b = w_b
    g  = g

    return p, v, q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    
    # YOUR CODE HERE
    R = q.as_matrix()
    a_skew = (a_m - a_b).flatten()
    a_skew = np.array([[0, -a_skew[2], a_skew[1]],
                       [a_skew[2], 0, -a_skew[0]],
                       [-a_skew[1], a_skew[0], 0]])
    
    omega_diff = w_m - w_b


    F_x = np.eye(18)
    F_x[:3,3:6] = np.eye(3)*dt
    F_x[3:6, 6:9] = -R @ a_skew * dt
    F_x[3:6, 9:12] = -R @ np.eye(3) * dt
    F_x[3:6, 15:18] = np.eye(3) * dt
    F_x[6:9, 6:9] = Rotation.from_rotvec((R.T @ (omega_diff*dt)).flatten()).as_matrix()
    F_x[6:9, 12:15] = -np.eye(3) * dt


    V_i = np.eye(3) * dt**2 * accelerometer_noise_density**2
    Theta_i = np.eye(3) * dt**2 * gyroscope_noise_density**2
    A_i = np.eye(3) * dt * accelerometer_random_walk**2
    Omega_i = np.eye(3) * dt * gyroscope_random_walk**2
    Q_i = np.eye(12)
    Q_i[:3, :3] = V_i
    Q_i[3:6, 3:6] = Theta_i
    Q_i[6:9, 6:9] = A_i
    Q_i[9:12, 9:12] = Omega_i

    F_i = np.eye(12)
    F_i = np.vstack([np.zeros(12),np.zeros(12),np.zeros(12),F_i,np.zeros(12),np.zeros(12),np.zeros(12)])

    P  = F_x @ error_state_covariance @ F_x.T + F_i @ Q_i @ F_i.T
    # return an 18x18 covariance matrix
    return P


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state
    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    R = q.as_matrix()
    Pc =  R.T @ (Pw-p)
    uv_hat = Pc[:2] / Pc[2]

    innovation = uv - uv_hat
    if np.linalg.norm(innovation) > error_threshold:
        return (p, v, q, a_b, w_b, g), error_state_covariance, innovation
    else: 
        dzdP = np.zeros((2, 3))
        dzdP[:2, :2] = np.eye(2)
        dzdP[0, 2] = -Pc[0,0]/Pc[2,0]
        dzdP[1, 2] = -Pc[1,0]/Pc[2,0]
        dzdP = dzdP / Pc[2,0]

        dpdTheta = np.array([[0, -Pc[2,0], Pc[1,0]],
                             [Pc[2,0], 0, -Pc[0,0]],
                             [-Pc[1,0], Pc[0,0], 0]])
        
        dpdP = -R.T

        dzdDeltaP = dzdP @ dpdP
        dzdDeltaTheta = dzdP @ dpdTheta
        H = np.zeros((2, 18))
        H[:, :3] = dzdDeltaP
        H[:, 6:9] = dzdDeltaTheta

        K = error_state_covariance @ H.T @ np.linalg.inv(H @ error_state_covariance @ H.T + Q)
        error_state_covariance = (np.eye(18) - K @ H) @ error_state_covariance @ (np.eye(18) - K @ H).T + K @ Q @ K.T
        delta_x = K @ innovation
        p = p + delta_x[:3]
        v = v + delta_x[3:6]
        q = Rotation.from_matrix(R @ Rotation.from_rotvec(delta_x[6:9].flatten()).as_matrix())
        a_b = a_b + delta_x[9:12]
        w_b = w_b + delta_x[12:15]
        g = g + delta_x[15:18]

        

    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation
