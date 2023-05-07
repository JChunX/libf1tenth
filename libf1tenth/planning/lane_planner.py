import numpy as np
from libf1tenth.planning import PathPlanner, Waypoints

class LanePlanner(PathPlanner):
    
    def __init__(self, lanes):
        '''
        LanePlanner selects a path based on 
        obstacle-free lanes
        
        Args:
        - lanes: list of (N, m) waypoint arrays, where N is the number of waypoints, ordered by priority
        '''
        super().__init__()
        self.lanes = lanes
        
    def plan(self, pose, occupancy_grid, current_lane):
        '''
        Plan a path through the waypoints given the occupancy grid.
        
        Args:
        - pose: Pose object
        - occupancy_grid: Occupancies object
        '''
        # check which lanes are obstacle-free
        lane_free = np.zeros(len(self.lanes), dtype=bool)
        
        for i, lane in enumerate(self.lanes):
            num_collisions, _ = Waypoints.check_collisions(
            lane[:,0], 
            lane[:,1], 
            pose, occupancy_grid,
            target_layer='obs',
            correct_offset=False)
            
            lane_free[i] = (num_collisions == 0)
            
        # select the first obstacle-free lane
        # if current lane is free, stay in current lane
        # if lane_free[current_lane]:
        #     lane_idx = current_lane
        # else:
        lane_idx = np.argmax(lane_free)
        lane = self.lanes[lane_idx]
        
        return lane, lane_idx, lane_free