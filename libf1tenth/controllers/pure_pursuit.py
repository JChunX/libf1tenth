import math
import time

import numpy as np

from libf1tenth.controllers import LateralController
from libf1tenth.filter import DerivativeFilter
from libf1tenth.planning.pose import Pose
from libf1tenth.util.quick_maths import nearest_point

class PurePursuitController(LateralController):
    
    def __init__(self, lookahead: float=1.5, kd_theta: float=0.1, wheelbase: float=0.33, buffer_size: int=5):
        '''
        pure persuit controller
        
        lookahead: waypoint lookahead 
        kd_theta: gain for heading error
        wheelbase: wheelbase of the vehicle

        modified: CV
        '''
        super().__init__()
        self.lookahead = lookahead
        self.kd_theta = kd_theta
        self.wheelbase = wheelbase
        self.theta_error_derivative_filter = DerivativeFilter(buffer_size=buffer_size)
        
    def _find_waypoint_to_track(self, pose, waypoints):
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

        Modified: CV, used the code in LQR
        '''
        position, theta = pose.position, pose.theta
        lookahead = self.lookahead
        front_axle_position = position + (self.wheelbase+lookahead)* np.array([math.cos(theta), math.sin(theta)])
        nearest_idx = nearest_point(front_axle_position[0], front_axle_position[1], waypoints)
        front_axle_pose = Pose.from_position_theta(front_axle_position[0], 
                                                   front_axle_position[1], 
                                                   theta)
        
        theta_ref = waypoints[nearest_idx, 3]
        theta_e = self._find_heading_error(theta, theta_ref)
        kappa = waypoints[nearest_idx, 4]
        velocity = waypoints[nearest_idx, 2]
        if waypoints.shape[1] == 6:
            gain = waypoints[nearest_idx, 5]
        else: 
            gain = 0.3
        
        return  nearest_idx, theta_e, theta_ref, kappa, gain, velocity
    
    def get_steering_angle(self, pose, waypoints):
        
        (self.nearest_idx,
         self.theta_e, 
         theta_ref,  
         kappa,
         gain, velocity) = self._find_waypoint_to_track(pose, waypoints)
        
        self.theta_error_derivative_filter.update(self.theta_e)
        waypoint_ego = self._waypoint_to_ego(pose, waypoints[self.nearest_idx])
        
        cur_theta_error_derivative = self.theta_error_derivative_filter.get_value()

        angle = gain * (2*(waypoint_ego[1]))/(self.lookahead ** 2)# + self.kd_theta * cur_theta_error_derivative
        
        return angle, waypoints[self.nearest_idx], velocity
