import numpy as np


class LateralController:
    
    def __init__(self, angle_min=np.deg2rad(-24), angle_max=np.deg2rad(24)):
        self.angle_min = angle_min
        self.angle_max = angle_max
        
    def _waypoint_to_ego(self, pose, waypoint):
        position = pose[:2]
        orientation = pose[2]
        # transform target waypoint to ego car frame
        RCM = np.array([[np.cos(orientation), -np.sin(orientation)], 
                        [np.sin(orientation), np.cos(orientation)]]).T
        waypoint_ego = RCM @ (waypoint[:2] - position)
        
        return waypoint_ego
        
    def get_steering_angle(self, pose, waypoints):
        '''
        gets the steering angle for the car to follow the waypoints
        
        pose: [x, y, theta]
        waypoints: ndarray of shape (N, 2)
        '''
        
        raise NotImplementedError
    
    def _safety_bound(self, angle):
        return np.clip(angle, self.angle_min, self.angle_max)