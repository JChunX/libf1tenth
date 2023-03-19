import numpy as np
from scipy.spatial.transform import Rotation as R


class Pose:
    
    def __init__(self, x, y, rotation, velocity=0.0):
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
        Transform a point from the global frame to the pose frame.
        
        point: a 2D point in the global frame
        """
        # pad point with 0 on z axis
        R_pose_to_global = self.rot_mat_2d.T
        point_pose_frame = R_pose_to_global @ (point - self.position)
        
        return point_pose_frame