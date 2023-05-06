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
        
    def plan(self, pose, occupancy_grid):
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
            target_layer='laser',
            correct_offset=False)
            
            lane_free[i] = (num_collisions == 0)
            
        # select the first obstacle-free lane
        lane_idx = np.argmax(lane_free)
        lane = self.lanes[lane_idx]
        
        return lane