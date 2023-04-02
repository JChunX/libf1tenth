import time

import numpy as np
from numba import njit

from libf1tenth.planning.graph import PlanGraph, PlanNode
from libf1tenth.planning.waypoints import Waypoints


class PathPlanner:
    '''
    PathPlanner is a base class for path planning algorithms.
    '''
    def __init__(self):
        pass
    
    def plan(self, occupancy_grid):
        '''
        Plan a path through the waypoints given the occupancy grid.
        
        waypoints: Waypoints object
        occupancy_grid: Occupancies object
        '''
        raise NotImplementedError