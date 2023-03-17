import numpy as np
from libf1tenth.controllers import LateralController
from libf1tenth.filter import DerivativeFilter

class StanleyController(LateralController):
    def __init__(self, K=1.0, Kd=0.1, wheelbase=0.58, lookahead=0.25):
        super().__init__()
        self.K = K
        self.Kd = Kd
        self.wheelbase = wheelbase
        self.lookahead = lookahead
        self.crosstrack_error = 0.0
        self.d_crosstrack_error = DerivativeFilter()
        self.d_crosstrack_error.update(0.0)
        
    def _find_waypoint_to_track(self, pose, steering, waypoints):
        position = pose[:2]
        heading = pose[2]
        
        com_to_front_axle_length = self.wheelbase / 2.0
        front_axle_position = position + com_to_front_axle_length * np.array([np.cos(heading), np.sin(heading)])
        front_axle_augmented = front_axle_position + self.lookahead * np.array([np.cos(heading + steering), np.sin(heading + steering)])
        
        # find the closest waypoint to the front axle
        crosstrack_waypoint_idx = np.argmin(np.linalg.norm(waypoints[:, :2] - front_axle_position, axis=1))
        front_axle_pose = np.array([front_axle_position[0], 
                                    front_axle_position[1], 
                                    heading + steering])
        # find the closest waypoint to the front axle augmented position
        crosstrack_waypoint_idx_augmented = np.argmin(np.linalg.norm(waypoints[:, :2] - front_axle_augmented, axis=1))
        
        crosstrack_error = self._find_crosstrack_error(front_axle_pose, waypoints, crosstrack_waypoint_idx)
        
        return crosstrack_waypoint_idx_augmented, crosstrack_error
    
    def _find_crosstrack_error(self, front_axle_pose, waypoints, waypoint_idx):
        waypoint = waypoints[waypoint_idx]
        waypoint_ego = self._waypoint_to_ego(front_axle_pose, waypoint)
        crosstrack_error = waypoint_ego[1]
        self.crosstrack_error = crosstrack_error
        
        return crosstrack_error
    
    def _get_steering_error(self, pose, steering, waypoints, waypoint_idx):
        heading = pose[2]
        steering_heading = heading + steering
        waypoint_heading = waypoints[waypoint_idx, 3]
        
        steering_error = waypoint_heading - steering_heading
        steering_error = np.arctan2(np.sin(steering_error), np.cos(steering_error)) # normalize to [-pi, pi]
        
        return steering_error
        
    def get_steering_angle(self, pose, waypoints, steering=0.0, velocity=0.0):
        
        waypoint_idx, crosstrack_error = self._find_waypoint_to_track(pose, steering, waypoints)
        steering_error = self._get_steering_error(pose, steering, waypoints, waypoint_idx)
        self.d_crosstrack_error.update(crosstrack_error)
        if velocity > 2.0:
            angle = steering_error + np.arctan2(self.K * crosstrack_error + 
                                                self.Kd * self.d_crosstrack_error.get_value(),
                                                velocity)
        else:
            angle = steering_error
        angle = self._safety_bound(angle)
        
        return angle, waypoints[waypoint_idx]