import numpy as np
from scipy.interpolate import CubicSpline


class FrenetFrame:
    
    def __init__(self, waypoints):
        '''
        Defines a frenet frame for a given set of waypoints
        
        Args:
        - waypoints: (n, 5) array of waypoints [x, y, velocity, yaw, curvature]
        '''
        self.x = waypoints[:,0]
        self.y = waypoints[:,1]
        self.velocity = waypoints[:,2]
        self.yaw = waypoints[:,3]
        self.curvature = waypoints[:,4]
        
        # path progress
        self.s = np.zeros_like(self.x)
        self.s[1:] = np.cumsum(np.sqrt(np.diff(self.x)**2 + np.diff(self.y)**2))
        self.s_max = self.s[-1]
        
        # parametric splines
        self.cs_sx = CubicSpline(self.s, self.x)
        self.cs_sy = CubicSpline(self.s, self.y)
        self.cs_syaw = CubicSpline(self.s, self.yaw)
        
    def progress_diff(self, s0, s1):
        '''
        Calculate progress difference between two points
        Should take wraparound into account
        
        Args:
        - s0: prev path progress, float
        - s1: next path progress, float
        
        eg.
        s_max = 60
        s0 = 1, s1 = 4 -> diff = 3
        s0 = 59, s1 = 1 -> diff = 2
        '''
        diff = s1 - s0
        if diff < 0:
            diff += self.s_max
        return diff
        
    def frenet_to_cartesian(self, s, d):
        '''
        Converts points in frenet frame to cartesian frame
        
        Args:
        - s: path progress, ndarray or float
        - d: lateral offset, ndarray or float
        
        Returns:
        - x: x coordinate, ndarray or float
        - y: y coordinate, ndarray or float
        '''
        x = self.cs_sx(s) - d * np.sin(self.cs_syaw(s))
        y = self.cs_sy(s) + d * np.cos(self.cs_syaw(s))
        
        return x, y
    
    def cartesian_to_frenet(self, x, y):
        '''
        Converts points in cartesian frame to frenet frame
        
        Args:
        - x: x coordinate, ndarray or float
        - y: y coordinate, ndarray or float
        
        Returns:
        - s: path progress, ndarray or float
        - d: lateral offset, ndarray or float
        '''
        # if x is scalar, make it a 1d array
        is_scalar = np.isscalar(x)
        if is_scalar:
            x = np.array([x])
            y = np.array([y])
        
        dists = np.sqrt((self.x[:,None] - x)**2 + (self.y[:,None] - y)**2)

        closest_idx = np.argmin(dists, axis=0)
        prev_idx = closest_idx - 2
        s = self.s[closest_idx]
        s_prev = self.s[prev_idx]
        d = dists[closest_idx, np.arange(len(x))] * np.sign(np.cross(
            np.hstack(((self.cs_sx(s) - self.cs_sx(s_prev)).reshape(-1,1), 
                    (self.cs_sy(s) - self.cs_sy(s_prev)).reshape(-1,1))), 
            np.hstack(((x - self.cs_sx(s)).reshape(-1,1), 
                    (y - self.cs_sy(s)).reshape(-1,1)))))
        
        if is_scalar:
            s = s[0]
            d = d[0]
            
        return s, d