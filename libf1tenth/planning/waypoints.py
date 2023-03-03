import pandas as pd 
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import interpolate
from scipy.ndimage import gaussian_filter


class Waypoints:
             
    def __init__(self, x, y, steering, velocity, yaw, yaw_rate, slip_angle, is_periodic=False):
        self.x = x.astype(np.double)
        self.y = y.astype(np.double)
        self.steering = steering
        self.velocity = velocity
        self.yaw = yaw
        self.yaw_rate = yaw_rate
        self.slip_angle = slip_angle
        self.is_periodic = is_periodic
        self.create_splines()
        
    @classmethod
    def from_csv(cls, path, is_periodic=False):
        df = pd.read_csv(path)
        return cls(df['x'].values, df['y'].values, df['steering'].values, df['velocity'].values, df['yaw'].values, df['yaw_rate'].values, df['slip_angle'].values, is_periodic=is_periodic)
    
    @classmethod
    def from_numpy(cls, arr):
        assert arr.shape[1] == 7, "array must have 7 columns"
        return cls(arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4], arr[:,5], arr[:,6])
        
    def create_splines(self):
        t = np.arange(len(self.x))
        if self.is_periodic:
            #t_2 = np.arange(len(self.x), 2*len(self.x)-1)
            #self.cs_x = CubicSpline(np.concatenate((t, t_2)), 
            #                        np.concatenate((self.x, self.x[1:])), 
            #                        bc_type='periodic')
            #self.cs_y = CubicSpline(np.concatenate((t, t_2)), 
            #                        np.concatenate((self.y, self.y[1:])), 
            #                        bc_type='periodic')
            self.cs_x = CubicSpline(t, self.x, bc_type='periodic')
            self.cs_y = CubicSpline(t, self.y, bc_type='periodic')
        else:
            self.cs_x = CubicSpline(t, self.x)
            self.cs_y = CubicSpline(t, self.y)
        self.s_steering = interpolate.interp1d(t, self.steering, kind='linear', fill_value='extrapolate')
        self.s_velocity = interpolate.interp1d(t, self.velocity, kind='linear', fill_value='extrapolate')
        self.s_yaw = interpolate.interp1d(t, self.yaw, kind='linear', fill_value='extrapolate')
        self.s_yaw_rate = interpolate.interp1d(t, self.yaw_rate, kind='linear', fill_value='extrapolate')
        self.s_slip_angle = interpolate.interp1d(t, self.slip_angle, kind='linear', fill_value='extrapolate')
        
    def upsample(self, factor):
        # assert factor is an integer
        assert factor == int(factor), "factor must be an integer"

        ts = np.arange(0, len(self.x), 1/factor)[:-factor+1]
        x = self.cs_x(ts)
        y = self.cs_y(ts)
        
        if self.is_periodic:
            x[-1] = x[0]
            y[-1] = y[0]

        steering = self.s_steering(ts)
        # convolve steering, with same padding
        kernel_size = 3
        kernel = np.ones(kernel_size)/kernel_size
        steering = np.convolve(steering, kernel, mode='same')
        
        velocity = self.s_velocity(ts)
        yaw = self.s_yaw(ts)
        yaw_rate = self.s_yaw_rate(ts)
        slip_angle = self.s_slip_angle(ts)
        return Waypoints(x, y, steering, velocity, yaw, yaw_rate, slip_angle)
    
    def smooth(self, sigma):
        x = gaussian_filter(np.concatenate((self.x, self.x, self.x)), sigma=sigma)[len(self.x):2*len(self.x)]
        y = gaussian_filter(np.concatenate((self.y, self.y, self.y)), sigma=sigma)[len(self.y):2*len(self.y)]

        return Waypoints(x, y, 
                         self.steering, 
                         self.velocity, 
                         self.yaw, 
                         self.yaw_rate, 
                         self.slip_angle)
        
        
    def to_csv(self, csv):
        df = pd.DataFrame({'x': self.x, 'y': self.y, 'steering': self.steering, 'velocity': self.velocity, 'yaw': self.yaw, 'yaw_rate': self.yaw_rate, 'slip_angle': self.slip_angle})
        df.to_csv(csv, index=False)
        
    def to_numpy(self):
        return np.hstack((self.x[:,None], self.y[:,None], self.steering[:,None], self.velocity[:,None], self.yaw[:,None], self.yaw_rate[:,None], self.slip_angle[:,None]))
        
    def __len__(self):
        return self.x.shape[0]