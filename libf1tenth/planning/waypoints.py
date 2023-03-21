import pandas as pd 
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import interpolate
from scipy.ndimage import gaussian_filter

class Waypoints:
    '''
    Waypoints
    
    waypoints are a set of x, y coordinates and velocities
    upon initialization, splines are created for upsampling, yaw, and curvature
    '''
             
    def __init__(self, x, y, velocity, is_periodic=False):
        self.t = np.arange(len(x))
        self.x = x.astype(np.double)
        self.y = y.astype(np.double)
        self.velocity = velocity
        self.is_periodic = is_periodic
        self._create_splines()
        
    @property
    def yaw(self):
        return self._compute_yaw()
    
    @property
    def curvature(self):
        return self._compute_curvature()
        
    @classmethod
    def from_csv(cls, path, is_periodic=False):
        df = pd.read_csv(path)
        return cls(df['x'].values, 
                   df['y'].values, 
                   df['velocity'].values, 
                   is_periodic=is_periodic)
    
    @classmethod
    def from_numpy(cls, arr):
        assert arr.shape[1] >= 3, "array must have >= 3 columns"
        return cls(arr[:,0], arr[:,1], arr[:,2])
    
    def _compute_yaw(self):
        t = self.t
        yaw = np.arctan2(self.cs_dy(t), self.cs_dx(t))
        return yaw
        
    def _compute_curvature(self):
        t = self.t
        dx = self.cs_dx(t)
        dy = self.cs_dy(t)
        ddx = self.cs_ddx(t)
        ddy = self.cs_ddy(t)
        
        curvature = (dx*ddy - dy*ddx) / (dx**2 + dy**2) ** (3/2)
        return curvature

    def _create_splines(self):
        t = self.t
        if self.is_periodic:
            self.cs_x = CubicSpline(t, self.x, bc_type='periodic')
            self.cs_y = CubicSpline(t, self.y, bc_type='periodic')
        else:
            self.cs_x = CubicSpline(t, self.x)
            self.cs_y = CubicSpline(t, self.y)
            
        self.cs_dx = self.cs_x.derivative()
        self.cs_dy = self.cs_y.derivative()
        self.cs_ddx = self.cs_dx.derivative()
        self.cs_ddy = self.cs_dy.derivative()
        
        self.s_velocity = interpolate.interp1d(t, 
                            self.velocity, 
                            kind='linear', 
                            fill_value='extrapolate')
        
    def upsample(self, factor):
        # assert factor is an integer
        assert factor == int(factor), "factor must be an integer"

        ts = np.arange(0, len(self.x), 1/factor)[:-factor+1]
        x = self.cs_x(ts)
        y = self.cs_y(ts)
        
        if self.is_periodic:
            x[-1] = x[0]
            y[-1] = y[0]
        
        velocity = self.s_velocity(ts)

        return Waypoints(x, y, velocity)
    
    def smooth(self, sigma):
        x = gaussian_filter(self.x, sigma=sigma)
        y = gaussian_filter(self.y, sigma=sigma)

        return Waypoints(x, y,  
                         self.velocity)
        
    def to_csv(self, csv):
        df = pd.DataFrame({'x': self.x, 
                           'y': self.y, 
                           'velocity': self.velocity,
                           'yaw': self.yaw,
                           'curvature': self.curvature})
        df.to_csv(csv, index=False)
        
    def to_numpy(self):
        return np.hstack((self.x[:,None], 
                          self.y[:,None], 
                          self.velocity[:,None],
                          self.yaw[:,None],
                          self.curvature[:,None]))
    
    def __getitem__(self, idx):
        return (self.x[idx], 
                self.y[idx], 
                self.velocity[idx], 
                self.yaw[idx], 
                self.curvature[idx])
        
    def __len__(self):
        return self.x.shape[0]