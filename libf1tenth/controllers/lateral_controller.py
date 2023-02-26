import numpy as np


class LateralController:
    
    def __init__(self):
        pass
    
    def get_steering_angle(self, pose, waypoints):
        '''
        gets the steering angle for the car to follow the waypoints
        
        pose: [x, y, theta]
        waypoints: ndarray of shape (N, 2)
        '''
        
        raise NotImplementedError