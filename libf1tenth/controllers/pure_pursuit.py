import numpy as np
from .lateral_controller import LateralController


class PurePursuitController(LateralController):
    
    def __init__(self, K, lookahead):
        super().__init__()
        self.K = K
        self.lookahead = lookahead
        
    def _find_waypoint_to_track(self, pose, waypoints):
        """
        Find the closest waypoint to the car's current position
        
        pose: [x, y, theta]
        waypoints: ndarray of shape (N, 2)
        
        steps:
        1. if no waypoints within lookahead distance, return the closest waypoint
        2. if waypoints within lookahead distance, track the farthest waypoint within lookahead distance
        """
        
        position = pose[:2]
        heading = pose[2]
        distance_to_waypoints = np.linalg.norm(waypoints - position, axis=1)
        waypoint_indices_in_lookahead = np.where(distance_to_waypoints < self.lookahead)[0]
        
        waypoints_in_lookahead = waypoints[waypoint_indices_in_lookahead,:]
        
        # check if waypoints_in_lookahead is empty
        if waypoints_in_lookahead.size == 0:
            # return closest waypoint
            waypoint_to_track = waypoints[np.argmin(distance_to_waypoints)]
        
        # else, return the farthest waypoint within lookahead distance
        else:
            distance_to_waypoints_in_lookahead = distance_to_waypoints[waypoint_indices_in_lookahead]
            # augment distance by heading. points in front of the car have a lower distance
            heading_vector = np.array([np.cos(heading), np.sin(heading)])
            
            normalized_waypoint_headings = (waypoints_in_lookahead - position) / distance_to_waypoints_in_lookahead[:,None]
            
            augmented_distance_to_waypoints_in_lookahead = distance_to_waypoints_in_lookahead * np.dot(normalized_waypoint_headings, heading_vector)
            
            waypoint_to_track = waypoints_in_lookahead[np.argmax(augmented_distance_to_waypoints_in_lookahead)]

        return waypoint_to_track
    
    def get_steering_angle(self, pose, waypoints):
        position = pose[:2]
        orientation = pose[2]
        waypoint_to_track = self._find_waypoint_to_track(pose, waypoints)
        # transform target waypoint to ego car frame
        RCM = np.array([[np.cos(orientation), -np.sin(orientation)], 
                        [np.sin(orientation), np.cos(orientation)]]).T
        waypoint_ego = RCM @ (waypoint_to_track - position)
        angle = self.K * (2*(waypoint_ego[1]))/(self.lookahead ** 2)
        angle = np.clip(angle, np.deg2rad(-24), np.deg2rad(24))
        
        return angle, waypoint_to_track