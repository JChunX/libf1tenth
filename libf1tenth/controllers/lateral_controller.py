import numpy as np


class LateralController:
    
    def __init__(self, angle_min=np.deg2rad(-35), angle_max=np.deg2rad(35)):
        self.angle_min = angle_min
        self.angle_max = angle_max
        
    # TODO: get crosstrack error
        
    def get_steering_angle(self, pose, waypoints):
        '''
        gets the steering angle for the car to follow the waypoints
        
        pose: [x, y, theta]
        waypoints: ndarray of shape (N, 2)
        '''
        
        raise NotImplementedError
    
    def _safety_bound(self, angle):
        return np.clip(angle, self.angle_min, self.angle_max)