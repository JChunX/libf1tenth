import numpy as np
from libf1tenth.controllers import LateralController
from libf1tenth.filter import DerivativeFilter
from libf1tenth.planning.pose import Pose

class StanleyController(LateralController):
    def __init__(self, K=1.0, Kd=0.1, wheelbase=0.58, 
                 lookahead_schedule=([3.0, 4.0, 7.0],
                                     [0.8, 0.98, 1.9])): # .7 .97, 2.1
        super().__init__()
        self.K = K
        self.Kd = Kd
        self.wheelbase = wheelbase
        self.crosstrack_error = 0.0
        self.d_crosstrack_error = 0.0
        self.steering_error = 0.0
        self._d_crosstrack_error_filter = DerivativeFilter(buffer_size=5)
        self._d_crosstrack_error_filter.update(0.0)
        
        self.waypoint_idx = 0
        self.lookahead_waypoint_idx = 0
        
        self.velocity = 0.0
        self.curvature = 0.0
        self.lookahead_schedule = lookahead_schedule
        
    @property
    def lookahead(self):
        if self.velocity < self.lookahead_schedule[0][0]:
            lookahead = self.lookahead_schedule[1][0]
        elif self.velocity > self.lookahead_schedule[0][-1]:
            lookahead = self.lookahead_schedule[1][-1]
        else:
            lookahead = np.interp(self.velocity, self.lookahead_schedule[0], self.lookahead_schedule[1])
            
        return lookahead
        
    def _find_waypoint_to_track(self, pose, steering, waypoints):
        position = pose.position
        heading = pose.theta
        
        com_to_front_axle_length = self.wheelbase / 2.0
        front_axle_position = position + com_to_front_axle_length * np.array([np.cos(heading), np.sin(heading)])
        front_axle_augmented = front_axle_position + self.lookahead * np.array([np.cos(heading + steering), np.sin(heading + steering)])
        
        # find the closest waypoint to the front axle
        crosstrack_waypoint_idx = np.argmin(np.linalg.norm(waypoints[:, :2] - front_axle_position, axis=1))
        front_axle_pose = Pose.from_position_theta(
            front_axle_position[0], 
            front_axle_position[1], 
            heading+steering)

        # find the closest waypoint to the front axle augmented position
        crosstrack_waypoint_idx_augmented = np.argmin(np.linalg.norm(waypoints[:, :2] - front_axle_augmented, axis=1))
        
        self._find_crosstrack_error(front_axle_pose, waypoints, crosstrack_waypoint_idx)
        
        self.waypoint_idx = crosstrack_waypoint_idx
        self.lookahead_waypoint_idx = crosstrack_waypoint_idx_augmented
    
    def _find_crosstrack_error(self, front_axle_pose, waypoints, waypoint_idx):
        waypoint = waypoints[waypoint_idx]
        waypoint_ego = self._waypoint_to_ego(front_axle_pose, waypoint)
        crosstrack_error = waypoint_ego[1]
        self.crosstrack_error = crosstrack_error
    
    def _find_steering_error(self, heading, steering, waypoints, waypoint_idx):
        steering_heading = heading + steering
        waypoint_heading = waypoints[waypoint_idx, 3]
        
        steering_error = waypoint_heading - steering_heading
        self.steering_error = np.arctan2(np.sin(steering_error), np.cos(steering_error)) # normalize to [-pi, pi]
        
    def _find_curvature_augmentation_coefficient(self):
        curvature = self.curvature
        if self.velocity > 4.2 or curvature < 0.02: 
            K_curve = 1.0
        else:
            # clip curvature to [0, 0.5]
            curvature = np.clip(curvature, 0.0, 0.5)
            min_curvature = 0.0
            min_K_curve = 1.3 # 1.3
            max_curvature = 0.5
            max_K_curve = 3.6 # 3.6
            
            K_curve = np.interp(curvature, [min_curvature, max_curvature], [min_K_curve, max_K_curve])
        
        return K_curve
    
    def augment_K(self):
        curvature = self.curvature
        if self.velocity > 5.0:
            return self.K / 5.5, self.Kd / 1.8
        if self.velocity > 4.1:
            return self.K / 6, self.Kd / 1.8 #17,1.8
        if curvature < 0.02:
            return self.K / 6, self.Kd / 1.5 # kp3, kd1.2
        else:
            return self.K, self.Kd
    
    def get_steering_angle(self, pose, waypoints, steering=0.0):
        self.velocity = pose.velocity
        
        self._find_waypoint_to_track(pose, steering, waypoints)
        self._find_steering_error(pose.theta, steering, waypoints, self.lookahead_waypoint_idx)
        
        self._d_crosstrack_error_filter.update(self.crosstrack_error)
        self.d_crosstrack_error = self._d_crosstrack_error_filter.get_value()

        self.curvature = waypoints[self.waypoint_idx, 4]
        K_curve = self._find_curvature_augmentation_coefficient()
        K, Kd = self.augment_K()
        
        if self.velocity > 2.0:
            # angle = K_curve * self.steering_error + 4.0 * np.arctan2(
            #     self.K * self.crosstrack_error + 
            #     self.Kd * self.d_crosstrack_error,
            #     self.velocity
            # )
            arctan_term = np.arctan2(self.crosstrack_error, self.velocity)
            d_arctan_term = np.arctan2(self.d_crosstrack_error, self.velocity)
            angle = (K_curve * self.steering_error
                     + K * arctan_term 
                     + Kd * d_arctan_term) 
        else:
            angle = self.steering_error
        angle = self._safety_bound(angle)
        
        return angle, waypoints[self.lookahead_waypoint_idx]