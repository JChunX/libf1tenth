import pandas as pd 
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import interpolate


class Waypoints:
             
    def __init__(self, x, y, steering, velocity, yaw, yaw_rate, slip_angle):
        self.x = x.astype(np.double)
        self.y = y.astype(np.double)
        self.steering = steering
        self.velocity = velocity
        self.yaw = yaw
        self.yaw_rate = yaw_rate
        self.slip_angle = slip_angle
        self.create_splines()
        
    @classmethod
    def from_csv(cls, path):
        df = pd.read_csv(path)
        return cls(df['x'].values, df['y'].values, df['steering'].values, df['velocity'].values, df['yaw'].values, df['yaw_rate'].values, df['slip_angle'].values)
        
    def create_splines(self):
        t = np.arange(len(self.x))
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
        x[-1] = x[0]
        y = self.cs_y(ts)
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
        
    def to_csv(self, csv):
        df = pd.DataFrame({'x': self.x, 'y': self.y, 'steering': self.steering, 'velocity': self.velocity, 'yaw': self.yaw, 'yaw_rate': self.yaw_rate, 'slip_angle': self.slip_angle})
        df.to_csv(csv, index=False)
        
    def to_numpy(self):
        return np.hstack((self.x[:,None], self.y[:,None], self.steering[:,None], self.velocity[:,None], self.yaw[:,None], self.yaw_rate[:,None], self.slip_angle[:,None]))
        
    def __len__(self):
        return self.x.shape[0]