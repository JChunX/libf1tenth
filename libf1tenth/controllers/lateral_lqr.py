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
from libf1tenth.util.quick_maths import nearest_point, solve_lqr, linearized_discrete_lateral_dynamics
from typing import List

class LateralLQRController(LateralController):
    def __init__(self, 
                 control_vels: List[float]=[3.0, 3.5, 4.0, 5.0],
                 Qs = [[0.3,0.01,0.001,0.022],
                       [0.45,0.01,0.001,0.022],
                       [0.55,0.01,0.001,0.046],
                       [0.55,0.01,0.001,0.046]],
                 lookaheads = [0.2,0.3,0.35, 0.4],
                 R: List[float]=[0.0062], 
                 iterations: int=50, 
                 eps: float=0.01, 
                 wheelbase: float=0.33,
                 lookahead_slow: float=0.3,
                 lookahead_fast: float=0.5):
        super().__init__()
        
        self.R = np.diag(R)
        self.iterations = iterations
        self.eps = eps
        self.control_vels = control_vels
        self.Qs = np.array(Qs)
        
        self.wheelbase = wheelbase
        
        self.crosstrack_error = 0.0
        self.theta_e = 0.0
        self.d_crosstrack_error = DerivativeFilter(buffer_size=4)
        self.d_crosstrack_error.update(0.0)
        self.d_theta_e = DerivativeFilter(buffer_size=4)
        self.d_theta_e.update(0.0)
        
        self.nearest_idx = 0
        self.theta_ref = 0.0
        self.kappa_ref = 0.0
        
        self.dt = DerivativeFilter()
        self.prev_time = None
        
        self.next_pred_state = np.zeros((4, 1))
        self.lookaheads = lookaheads
    
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
        lookahead = self.get_lookahead(pose.velocity)
        front_axle_position = position + (self.wheelbase+lookahead)* np.array([math.cos(theta), math.sin(theta)])
        nearest_idx = nearest_point(front_axle_position[0], front_axle_position[1], waypoints)
        front_axle_pose = Pose.from_position_theta(front_axle_position[0], 
                                                   front_axle_position[1], 
                                                   theta)
        
        theta_ref = waypoints[nearest_idx, 3]
        theta_e = self._find_heading_error(theta, theta_ref)
        crosstrack_error = self._find_crosstrack_error(front_axle_pose, waypoints[nearest_idx])
        kappa_ref = waypoints[nearest_idx, 4]
        
        return theta_e, crosstrack_error, theta_ref, kappa_ref, nearest_idx
    
    def get_lookahead(self, velocity):
        if velocity < self.control_vels[0]:
            return self.lookaheads[0]
        if velocity > self.control_vels[1]:
            return self.lookaheads[1]
        
        return np.interp(velocity, self.control_vels, self.lookaheads)
        
    def get_Q(self, velocity):
        # interpolated q1, q2, q3, q4, then construct diag matrix
        if velocity < self.control_vels[0]:
            return np.diag(self.Qs[0])
        if velocity > self.control_vels[1]:
            return np.diag(self.Qs[1])
        
        q1s = self.Qs[:,0]
        q2s = self.Qs[:,1]
        q3s = self.Qs[:,2]
        q4s = self.Qs[:,3]
        
        q1 = np.interp(velocity, self.control_vels, q1s)
        q2 = np.interp(velocity, self.control_vels, q2s)
        q3 = np.interp(velocity, self.control_vels, q3s)
        q4 = np.interp(velocity, self.control_vels, q4s)
        
        return np.diag([q1, q2, q3, q4])
    
    def get_steering_angle(self, pose, waypoints):
        '''
        Compute the steering angle given the current pose and waypoints.
        
        Args:
        - pose: The current pose of the vehicle (Pose)
        - waypoints: (n, 5) ndarray of waypoints (x, y, theta, v, k)
        
        Returns:
        - steering_angle: The steering angle in radians (float)
        '''
        self.dt.update(time.time())
        
        (self.theta_e, 
         self.crosstrack_error, 
         self.theta_ref, 
         self.kappa_ref, 
         self.nearest_idx) = self._compute_control_points(pose, waypoints)
        
        self.d_crosstrack_error.update(self.crosstrack_error)
        self.d_theta_e.update(self.theta_e)
        
        dt = 0.01
        if self.dt.is_ready():
            dt = self.dt.get_value()

        state_size = 4
        Ad, Bd = linearized_discrete_lateral_dynamics(pose.velocity, state_size, dt, self.wheelbase)
        
        Q = self.get_Q()

        K = solve_lqr(Ad, Bd, Q, self.R, self.eps, self.iterations)

        state = np.zeros((state_size, 1))
        state[0][0] = self.crosstrack_error
        state[1][0] = self.d_crosstrack_error.get_value() / dt
        state[2][0] = self.theta_e
        state[3][0] = self.d_theta_e.get_value() / dt
        
        steer_angle_feedback = (K @ state)[0][0]
        pred_state_error = self.next_pred_state - state
        self.next_pred_state = (Ad @ state) + (Bd * steer_angle_feedback)

        #Compute feed forward control term to decrease the steady error.
        steer_angle_feedforward = self.kappa_ref * self.wheelbase

        # Calculate final steering angle in [rad]
        steer_angle = steer_angle_feedback + 1.0 * steer_angle_feedforward
        steer_angle = self._safety_bound(steer_angle)
        
        return steer_angle, waypoints[self.nearest_idx], pred_state_error.flatten(), state.flatten()
