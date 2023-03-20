import numpy as np
from scipy.spatial.transform import Rotation as R


class Pose:
    
    def __init__(self, x, y, rotation, velocity=0.0):
        '''
        Pose represents a vehicle pose in the global frame.
        
        Args:
        - x: x position in the global frame, positive x is forward
        - y: y position in the global frame, positive y is left
        - rotation: a scipy.spatial.transform.Rotation object representing the rotation of the vehicle
        - velocity: the longitudinal velocity of the vehicle, positive velocity is forward
        '''
        self.x = x
        self.y = y
        self.rotation = rotation
        self.theta = self.euler[2]
        self.velocity = velocity
        
    @classmethod
    def from_msg(cls, pose_msg):
        
        rotation = R.from_quat([pose_msg.pose.pose.orientation.x, 
                                pose_msg.pose.pose.orientation.y, 
                                pose_msg.pose.pose.orientation.z, 
                                pose_msg.pose.pose.orientation.w])
        return cls(pose_msg.pose.pose.position.x,
                   pose_msg.pose.pose.position.y,
                   rotation,
                   pose_msg.twist.twist.linear.x)
        
    @classmethod
    def from_position_theta(cls, x, y, theta, velocity=0.0):
        rotation = R.from_euler('xyz', [0, 0, theta])
        return cls(x, y, rotation, velocity)
        
    @property
    def position(self):
        return np.array([self.x, self.y])
    
    @property
    def quaternion(self):
        return self.rotation.as_quat()
    
    @property
    def euler(self):
        return self.rotation.as_euler('xyz')
    
    @property
    def rot_mat(self):
        return self.rotation.as_matrix()
    
    @property
    def rot_mat_2d(self):
        return self.rotation.as_matrix()[:2, :2]
        
    def __repr__(self):
        return f"Pose(x={self.x}, y={self.y}, theta={self.theta}, velocity={self.velocity})"
    
    def __eq__(self, other):
        return (self.x == other.x 
                and self.y == other.y 
                and self.theta == other.theta
                and self.velocity == other.velocity)
    
    def as_array(self):
        return np.array([self.x, self.y, self.theta, self.velocity])
    
    def global_point_to_pose_frame(self, point):
        """
        Transform a point from the global frame to this pose frame.
        
        Args:
        - point: a ndarray of shape (2,) representing a 2D point in the global frame
        
        Returns:
        - point_pose_frame: a ndarray of shape (2,) representing a 2D point in this pose frame
        """
        # pad point with 0 on z axis
        R_pose_to_global = self.rot_mat_2d.T
        point_pose_frame = R_pose_to_global @ (point - self.position)
        
        return point_pose_frame
    
    def pose_point_to_global_frame(self, point):
        """
        Transform a point from this pose frame to the global frame.
        
        Args:
        - point: a ndarray of shape (2,) representing a 2D point in this pose frame
        
        Returns:
        - point_global_frame: a ndarray of shape (2,) representing a 2D point in the global frame
        """
        R_pose_to_global = self.rot_mat_2d
        point_global_frame = R_pose_to_global @ point + self.position
        
        return point_global_frame
    