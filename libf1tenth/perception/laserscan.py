import numpy as np


'''
LaserScan message docs
           
float32 angle_min        # start angle of the scan [rad]
float32 angle_max        # end angle of the scan [rad]
float32 angle_increment  # angular distance between measurements [rad]

float32 time_increment   # time between measurements [seconds] - if your scanner
                         # is moving, this will be used in interpolating position
                         # of 3d points
float32 scan_time        # time between scans [seconds]

float32 range_min        # minimum range value [m]
float32 range_max        # maximum range value [m]

float32[] ranges         # range data [m]
'''

class Scan:
    
    def __init__(self, angle_min, angle_max, 
                 angle_increment, time_increment, 
                 scan_time, 
                 range_min, range_max, 
                 ranges):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = angle_increment
        self.time_increment = time_increment
        self.scan_time = scan_time
        self.range_min = range_min
        self.range_max = range_max
        self.ranges = ranges
        self.angles = np.arange(angle_min, angle_max, angle_increment)
    
    @classmethod
    def from_msg(cls, msg):
        return cls(msg.angle_min, msg.angle_max, 
                   msg.angle_increment, msg.time_increment, 
                   msg.scan_time, 
                   msg.range_min, msg.range_max, 
                   np.array(msg.ranges))
        
    def narrow_scan(self, ang_min, ang_max):
        """
        Narrows the scan to a given angular range.
        
        Args:
        - ang_min (float): the minimum angle of the narrowed scan in radians.
        - ang_max (float): the maximum angle of the narrowed scan in radians.
        """
        idx_min = int((ang_min - self.angle_min) // self.angle_increment)
        idx_max = int((ang_max - self.angle_min) // self.angle_increment)
        idx_min = max(0, idx_min)
        idx_max = min(len(self.ranges), idx_max)

        ranges = self.ranges[idx_min:idx_max]
        angles = self.angles[idx_min:idx_max]
        
        new_scan = Scan(ang_min, ang_max,
                        self.angle_increment, self.time_increment,
                        self.scan_time, self.range_min, self.range_max,
                        ranges)
        return new_scan
        
    def get_valid_ranges_and_angle(self):
        """
        Filters the invalid range measurements and their corresponding angles.
        
        Returns:
        - A tuple containing:
            - valid_ranges (ndarray): the array of valid range measurements in meters.
            - valid_angles (ndarray): the array of valid angles in radians corresponding to the valid range measurements.
        """
        idx_invalid_ranges = np.logical_or(
            np.logical_or(np.isnan(self.ranges), np.isinf(self.ranges)), 
            self.ranges > self.range_max)
        valid_ranges = self.ranges[~idx_invalid_ranges]
        valid_angles = self.angles[~idx_invalid_ranges]
        
        return valid_ranges, valid_angles
    
    def to_cartesian(self):
        """
        Converts the range measurements to Cartesian coordinates.
        
        Returns:
        - A tuple containing:
            - x (ndarray): the array of x coordinates in meters.
            - y (ndarray): the array of y coordinates in meters.
        """
        valid_ranges, valid_angles = self.get_valid_ranges_and_angle()
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        return x, y