import time

import numpy as np
from numba import njit

from libf1tenth.planning.graph import PlanGraph, PlanNode


class PathPlanner:
    '''
    PathPlanner is a base class for path planning algorithms.
    '''
    def __init__(self):
        pass
    
    def plan(self, waypoints, occupancy_grid):
        '''
        Plan a path through the waypoints given the occupancy grid.
        
        waypoints: Waypoints object
        occupancy_grid: Occupancies object
        '''
        raise NotImplementedError
    

class RRTPlanner(PathPlanner):
    '''
    RRT planner
    '''
    def __init__(self, search_radius, max_iterations, expansion_distance, goal_threshold, occ_grid_layer='laser'):
        '''
        search_radius: radius of the search circle
        max_iterations: maximum number of iterations
        expansion_distance: distance to expand the tree
        goal_threshold: distance to goal to consider goal reached
        '''
        self.search_radius = search_radius
        self.max_iterations = max_iterations
        self.expansion_distance = expansion_distance
        self.goal_threshold = goal_threshold
        
        self.occupancy_grid = None
        self.waypoints = None
        self.G = None
        
        self.occ_grid_layer = occ_grid_layer
        
    def plan(self, waypoints, occupancy_grid, start_pos):
        '''
        Plan a path through the waypoints given the occupancy grid.
        
        Args:
        - waypoints: Waypoints object
        - occupancy_grid: Occupancies object
        - start_pos: starting position in the ego frame
        '''
        self.waypoints = waypoints
        self.occupancy_grid = occupancy_grid
        
        self.G = PlanGraph(start_pos)
        
        for _ in range(self.max_iterations):
            # sample a point
            position = self._sample_free()
            new_node = PlanNode(position[0], position[1])
            
            nearest_node_id = self.G.get_nearest_node_idx(new_node)
            nearest_node = self.G.get_node(nearest_node_id)
            self._steer(new_node, nearest_node)
            
            if not self._check_collision(new_node, nearest_node):
                self.G.add_node(new_node)
                self.G.add_edge(nearest_node_id, new_node.id)
            
            # check if goal is reached
            #if self._is_goal_reached(new_node):
            #    break
    
    def _sample_free(self, layer='laser'):
        '''
        Sample a free point in the occupancy grid and get its position in the odom frame.
        
        Args:
        - occupancy_grid: Occupancies object
        - layer: layer of occupancy_grid to sample from
        
        Returns:
        - position: position of the sampled point in the odom frame
        '''
        occupancy = self.occupancy_grid.layers[layer]['occupancy']
        #free = np.argwhere(occupancy == 0)
        # sample a free point
        #free_x_idx, free_y_idx = free[np.random.randint(len(free))]
        
        free_x_idx, free_y_idx = RRTPlanner._get_free_idx(occupancy)
        
        x, y = self.occupancy_grid.grid_indices_to_pc(free_x_idx, free_y_idx)
        position = np.array([x, y])
        
        return position
    
    @staticmethod
    @njit
    def _get_free_idx(occupancy):
        free = np.argwhere(occupancy == 0)
        free_x_idx, free_y_idx = free[np.random.randint(len(free))]
        return free_x_idx, free_y_idx
    
    def _is_goal_reached(self, node):
        '''
        Check if the goal is reached.
        
        Args:
        - node: PlanNode object
        
        Returns:
        - is_goal_reached: boolean
        '''
        # TODO: implement this method
        pass
    
    def _steer(self, sampled_node, nearest_node):
        '''
        This method should modify the sampled_node such that it is closer 
        to the nearest_node than it was before.
        
        Args:
        - sampled_node: PlanNode object, sampled node
        - nearest_node: PlanNode object, nearest node to the sampled node
        '''
        sampled_node_x, sampled_node_y = RRTPlanner._steer_helper(
            nearest_node.x, nearest_node.y, 
            sampled_node.x, sampled_node.y, 
            self.expansion_distance
        )

        sampled_node.x = sampled_node_x
        sampled_node.y = sampled_node_y
        
    @staticmethod
    @njit
    def _steer_helper(nearest_node_x, nearest_node_y, sampled_node_x, sampled_node_y, expansion_distance):
        dx = nearest_node_x - sampled_node_x
        dy = nearest_node_y - sampled_node_y

        ang = np.arctan2(dy, dx)
        dx_new = expansion_distance * np.cos(ang)
        dy_new = expansion_distance * np.sin(ang)

        sampled_node_x = dx_new + nearest_node_x
        sampled_node_y = dy_new + nearest_node_y
        
        return sampled_node_x, sampled_node_y
        
    def _check_collision(self, sampled_node, nearest_node):
        '''
        Check if the line between the sampled node and the nearest node 
        intersects with any obstacles.
        
        Args:
        - sampled_node: PlanNode object, sampled node
        - nearest_node: PlanNode object, nearest node to the sampled node
        
        Returns:
        - is_collision: boolean
        '''
        # check if the line between the sampled node and the nearest node intersects with any obstacles
        x1, y1 = sampled_node.position
        x2, y2 = nearest_node.position
        is_collision = self.occupancy_grid.check_line_collision(self.occ_grid_layer, x1, y1, x2, y2)
        
        return is_collision
        
class RRTStarPlanner(RRTPlanner):
    '''
    RRT* planner
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def plan(self, waypoints, occupancy_grid):
        pass