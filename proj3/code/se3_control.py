import numpy as np
from scipy.spatial.transform import Rotation


class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        k = self.k_drag/self.k_thrust
        L = self.arm_length
        self.to_TM = np.array([[1,  1,  1,  1],
                               [ 0,  L,  0, -L],
                               [-L,  0,  L,  0],
                               [ k, -k,  k, -k]])
        self.weight = np.array([0, 0, -self.mass*self.g])
        self.rAB = np.eye(3)
        
        self.K_p = np.diag([7.5, 7.5, 8])  # Position gains
        self.K_d = np.diag([5, 5, 5])  # Velocity damping
        self.K_R = np.diag([2500, 2500, 200])  # Attitude gains
        self.K_w = np.diag([250, 250, 150])   # Angular rate damping




    def posControl(self,traj, state):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            -u1, the sum of the forces
            -rot_des, the desired orientation
        """
        r_ddot_des = traj['x_ddot'] - (self.K_d @ (state['v'] - traj['x_dot'])) - (self.K_p @ (state['x'] - traj['x']))
        fDes = self.mass * r_ddot_des + -self.weight
        self.rotAB = Rotation.from_quat(state['q']).as_matrix()
        u1 = self.rotAB[:,-1].T @ fDes

        #########R_Des Calculation
        f_des_norm = np.linalg.norm(fDes)
        if f_des_norm < 1e-6:  # Avoid division by zero
            b_3 = np.array([0, 0, 1])  # Default to upright thrust direction
        else:
            b_3 = fDes / f_des_norm

        a_yaw = np.array([np.cos(traj['yaw']),np.sin(traj['yaw']),0])
        b_2 = np.cross(b_3,a_yaw)/np.linalg.norm(np.cross(b_3,a_yaw))

        rot_des = np.array([np.cross(b_2,b_3),b_2,b_3]).T
        return u1, rot_des


    def attitudeControl(self,rot_des,state):

        e_R = 1/2 * ((rot_des.T @ self.rotAB) -(self.rotAB.T @ rot_des))
        e_R = np.array([e_R[2,1],e_R[0,2],e_R[1,0]])

        u2 = self.inertia @ (-self.K_R @ e_R - self.K_w @ state['w'])
        return u2

    def calcSpeeds(self,u1,u2):
        u = np.insert(u2,0,u1)
        forces = np.linalg.inv(self.to_TM) @ u
        forces /= self.k_thrust
        cmdSpeeds = np.clip(np.sqrt(np.maximum(forces,0)),self.rotor_speed_min,self.rotor_speed_max)
        return cmdSpeeds

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        

        u1,rot_des = self.posControl(flat_output,state)
        u2 = self.attitudeControl(rot_des,state)
        cmd_motor_speeds = self.calcSpeeds(u1,u2)
        cmd_thrust = u1
        cmd_moment = u2
        cmd_q = Rotation.from_matrix(rot_des).as_quat()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        
        return control_input

