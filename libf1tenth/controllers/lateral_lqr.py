import numpy as np

from libf1tenth.controllers import LateralController
from libf1tenth.filter import DerivativeFilter
from libf1tenth.planning.pose import Pose


class LateralLQRController(LateralController):
    def __init__(self, Q=[1.0, 0.95, 0.0066, 0.0257], 
                 R=[0.0062], 
                 iterations=50, 
                 eps=0.01, 
                 wheelbase=0.33):
        super().__init__()
        self.Q = Q
        self.R = R
        self.iterations = iterations
        self.eps = eps
        self.wheelbase = wheelbase
        self.crosstrack_error = 0.0
        self.d_crosstrack_error = DerivativeFilter()
        self.d_crosstrack_error.update(0.0)
        pass # TODO 
    
    def get_steering_angle(self, pose, waypoints):
        '''
        Compute the steering angle given the current pose and waypoints.
        
        Args:
        - pose: The current pose of the vehicle (Pose)
        - waypoints: (n, 5) ndarray of waypoints (x, y, theta, v, k)
        
        Returns:
        - steering_angle: The steering angle in radians (float)
        '''
        