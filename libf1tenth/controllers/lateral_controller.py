import math

import numpy as np

from libf1tenth.planning.pose import Pose


class LateralController:
    
    def __init__(self, angle_min=np.deg2rad(-35), angle_max=np.deg2rad(35)):
        self.angle_min = angle_min
        self.angle_max = angle_max
        
    def _waypoint_to_ego(self, pose, waypoint):
        return pose.global_position_to_pose_frame(waypoint[:2])
    
    def _find_heading_error(self, heading, waypoint_heading):
        '''
        Finds the heading error for the car given the current heading and target waypoint
        
        Args:
        - heading: float
        - waypoint_heading: float
        
        Returns:
        - heading_error: float
        '''
        
        heading_error = waypoint_heading - heading
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        return heading_error
    
    def _find_crosstrack_error(self, pose, waypoint):
        '''
        Finds the crosstrack error for the car given the current pose and target waypoint
        
        Args:
        - pose: Pose object
        - waypoint: (5, ) ndarray of (x, y, theta, heading, curvature)
        
        Returns:
        - crosstrack_error: float
        '''
        waypoint_ego = self._waypoint_to_ego(pose, waypoint)
        crosstrack_error = waypoint_ego[1]
        return crosstrack_error
        
    def get_steering_angle(self, pose, waypoints):
        '''
        Gets the steering angle for the car to follow the waypoints
        
        Args:
        - pose: Pose object
        - waypoints: ndarray of shape (N, 5) where each row is (x, y, theta, heading, curvature)
        
        Returns:
        - steering_angle: float
        '''
        
        raise NotImplementedError
    
    def _safety_bound(self, angle):
        return max(self.angle_min, min(self.angle_max, angle))