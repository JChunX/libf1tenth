import numpy as np
from scipy.interpolate import CubicSpline
from numba import njit
from libf1tenth.util.quick_maths import l2_norm

class FrenetFrame:
    
    def __init__(self, waypoints, make_cs=True):
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
        self.make_cs = make_cs
        if self.make_cs:
            self.cs_sx = CubicSpline(self.s, self.x)
            self.cs_sy = CubicSpline(self.s, self.y)
            self.cs_syaw = CubicSpline(self.s, self.yaw)
            self.cs_sk = CubicSpline(self.s, self.curvature)
        
    def wrap(self, s):
        '''
        Wraps path progress to [0, s_max]
        
        Args:
        - s: path progress, ndarray or float
        '''
        return s % self.s_max
        
    def wrapped_diff(self, s0, s1):
        '''
        Calculate progress difference between two points
        Should take wraparound into account
        
        Args:
        - s0: prev path progress
        - s1: next path progress
        
        eg.
        s_max = 60
        s0 = 1, s1 = 4 -> diff = 3
        s0 = 59, s1 = 1 -> diff = 2
        '''
        
        is_scalar = np.isscalar(s0)
        if is_scalar:
            s0 = np.array([s0])
        
        diff = s1 - s0
        diff[diff > self.s_max/2] -= self.s_max
        diff[diff < -self.s_max/2] += self.s_max
        
        return diff[0] if is_scalar else diff
    
    def frenet_distance(self, position, other_positions):
        '''
        Computes frenet distance between a position and a set of other positions
        
        Args:
        - position (2,): (s0, d0) position to compute distance from
        - other_positions (n, 2) or (2,): (s, d) positions to compute distance to
        
        Returns:
        - dists (n,) or float: frenet distance to each other position
        '''
        is_1d = (other_positions.ndim == 1)
        if is_1d:
            other_positions = other_positions.reshape(1,-1)
        
        dists = l2_norm(self.wrapped_diff(other_positions[:,0], position[0]),
                       (other_positions[:,1] - position[1]))
        
        return dists[0] if is_1d else dists
    
    
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
        is_scalar = np.isscalar(s)
        if is_scalar:
            s = np.array([s])
            d = np.array([d])
        
        x_cart, y_cart = self.frenet_to_cartesian_numba(s, d, self.s, self.x, self.y, self.yaw)
        
        if is_scalar:
            x_cart = x_cart[0]
            y_cart = y_cart[0]
            
        return x_cart, y_cart
    
    @staticmethod
    @njit(cache=True, fastmath=True)  
    def frenet_to_cartesian_numba(s0, d0, s, x, y, yaw):
            
        x_cart = np.zeros_like(s0)
        y_cart = np.zeros_like(s0)
        for i in range(len(s0)):
            idx = np.argmin(np.abs(s - s0[i]))
            x_interp = x[idx]
            y_interp = y[idx]
            yaw_interp = yaw[idx]
            x_cart[i] = x_interp - d0[i] * np.sin(yaw_interp)
            y_cart[i] = y_interp + d0[i] * np.cos(yaw_interp)
            
        return x_cart, y_cart
    
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
        assert self.make_cs, 'Must have splines to convert to frenet frame, re-initialize with make_cs=True'
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