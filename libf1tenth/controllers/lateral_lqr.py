'''
LQR Lateral Controller

Reference:
https://github.com/f1tenth/f1tenth_planning/blob/main/f1tenth_planning/control/lqr/lqr.py
'''

import math
import time

import numpy as np

from libf1tenth.controllers import LateralController
from libf1tenth.filter import DerivativeFilter
from libf1tenth.planning.pose import Pose
from libf1tenth.util.fast_math import nearest_point, solve_lqr, update_matrix


class LateralLQRController(LateralController):
    def __init__(self, Q=[1.0, 0.95, 0.0066, 0.0257], 
                 R=[0.0062], 
                 iterations=50, 
                 eps=0.01, 
                 wheelbase=0.33):
        super().__init__()
        
        self.Q = np.diag(Q)
        self.R = np.diag(R)
        self.iterations = iterations
        self.eps = eps
        
        self.wheelbase = wheelbase
        
        self.crosstrack_error = 0.0
        self.theta_e = 0.0
        self.d_crosstrack_error = DerivativeFilter()
        self.d_crosstrack_error.update(0.0)
        self.d_theta_e = DerivativeFilter()
        self.d_theta_e.update(0.0)
        
        self.nearest_idx = 0
        self.theta_ref = 0.0
        self.kappa_ref = 0.0
        
        self.dt = DerivativeFilter()
        self.dt.update(0.01)
        self.prev_time = None
    
    def _compute_control_points(self, pose, waypoints):
        '''
        Compute the control points given the current pose and waypoints.
        
        Args:
        - pose: The current pose of the vehicle (Pose)
        - waypoints: (n, 5) ndarray of waypoints (x, y, v, yaw, k)
        
        Returns:
        - theta_e: Heading error (float)
        - crosstrack_error: Crosstrack error (float)
        - theta_ref: Target heading (float)
        - kappa_ref: Target curvature (float)
        - nearest_idx: Index of the nearest waypoint (int)
        '''
        position, theta = pose.position, pose.theta
        front_axle_position = position + self.wheelbase * np.array([math.cos(theta), math.sin(theta)])
        nearest_idx = nearest_point(front_axle_position[0], front_axle_position[1], waypoints)
        front_axle_pose = Pose.from_position_theta(front_axle_position[0], 
                                                   front_axle_position[1], 
                                                   theta)
        
        theta_ref = waypoints[nearest_idx, 3]
        theta_e = self._find_heading_error(theta, theta_ref)
        crosstrack_error = self._find_crosstrack_error(front_axle_pose, waypoints[nearest_idx])
        kappa_ref = waypoints[nearest_idx, 4]
        
        return theta_e, crosstrack_error, theta_ref, kappa_ref, nearest_idx
    
    def get_steering_angle(self, pose, waypoints):
        '''
        Compute the steering angle given the current pose and waypoints.
        
        Args:
        - pose: The current pose of the vehicle (Pose)
        - waypoints: (n, 5) ndarray of waypoints (x, y, theta, v, k)
        
        Returns:
        - steering_angle: The steering angle in radians (float)
        '''
        self.avg_dt.update(time.time())
        
        (self.theta_e, 
         self.crosstrack_error, 
         self.theta_ref, 
         self.kappa_ref, 
         self.nearest_idx) = self._compute_control_points(pose, waypoints)
        
        self.d_crosstrack_error.update(self.crosstrack_error)
        self.d_theta_e.update(self.theta_e)
        dt = self.avg_dt.get_value()

        state_size = 4
        Ad, Bd = update_matrix(pose, state_size, dt, self.wheelbase)
        K = solve_lqr(Ad, Bd, self.Q, self.R, self.eps, self.max_iterations)

        state = np.zeros((state_size, 1))
        state[0][0] = self.crosstrack_error
        state[1][0] = self.d_crosstrack_error.get_value() / dt
        state[2][0] = self.theta_e
        state[3][0] = self.d_theta_e.get_value() / dt
        
        steer_angle_feedback = (K @ state)[0][0]

        #Compute feed forward control term to decrease the steady error.
        steer_angle_feedforward = self.kappa_ref * self.wheelbase

        # Calculate final steering angle in [rad]
        steer_angle = steer_angle_feedback + steer_angle_feedforward

        return steer_angle
