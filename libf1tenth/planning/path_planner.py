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
        self.target_waypoint = None
        self.G = None
        self.destination_node_id = None
        self.is_goal_reached = False
        
        self.occ_grid_layer = occ_grid_layer
        
        
    def plan(self, waypoints, occupancy_grid, pose, start_pos=(0.,0.)):
        '''
        Plan a path through the waypoints given the occupancy grid.
        
        Args:
        - waypoints: ndarray (n, 5) waypoints
        - occupancy_grid: Occupancies object
        - pose: Pose object, current pose in global frame
        - start_pos: starting position in the ego frame, default is (0,0)
        '''
        

        self.waypoints = waypoints
        self.occupancy_grid = occupancy_grid
        lookahead = self.occupancy_grid.lookahead_distance
        
        candidate_goal_point = np.array([pose.x + lookahead * np.cos(pose.theta), 
                                         pose.y + lookahead * np.sin(pose.theta)])
        
        # find waypoint closest to the candidate goal point
        target_waypoint_idx = np.argmin(np.linalg.norm(self.waypoints[:, :2] - candidate_goal_point, axis=1))
        self.target_waypoint = waypoints[target_waypoint_idx, :2]
        self.target_waypoint = pose.global_position_to_pose_frame(self.target_waypoint)
        
        self.G = PlanGraph(start_pos)
        
        self.is_goal_reached = False
        for _ in range(self.max_iterations):
            # sample a point
            position = self._sample_free()
            new_node = PlanNode(position[0], position[1])
            
            nearest_node_id = self.G.get_nearest_node_idx(new_node)
            nearest_node = self.G.get_node(nearest_node_id)
            self._steer(new_node, nearest_node)
            
            if not self._check_collision(new_node, nearest_node):
                self.G.add_node(new_node)
                self.G.add_edge(nearest_node_id, new_node.id, parent_id=nearest_node_id)
            else:
                continue
            
            if self._is_goal_reached(new_node):
                self.destination_node_id = new_node.id
                self.is_goal_reached = True
                break
        
        return self.is_goal_reached
            
    def get_rrt_waypoints(self, pose, velocity=0.5, logger=None):
        '''
        Get the planned waypoints in the global frame.
        
        Args:
        - pose: Pose object, current pose in global frame
        '''
        if not self.is_goal_reached:
            return None

        destination_node = self.G.nodes[self.destination_node_id]
        positions_to_track = destination_node.position
        
        cur_node = destination_node
        while cur_node.parent is not None:
            cur_node = self.G.nodes[cur_node.parent.id]
            positions_to_track = np.vstack((cur_node.position, positions_to_track)) # shape (n, 2)
        
        if len(positions_to_track.shape) == 1:
            return None
        
        # convert to global frame
        positions_to_track = pose.pose_position_to_global_frame(positions_to_track.T).T
        
        waypoints_to_track = np.hstack((positions_to_track, 
                                        velocity * np.ones((positions_to_track.shape[0], 1))))
        
        waypoints = Waypoints.from_numpy(waypoints_to_track).upsample(50)#.smooth(1)
        
        return waypoints
    
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
        goal_x, goal_y = self.target_waypoint
        return np.linalg.norm(np.array([node.x - goal_x, node.y - goal_y])) < self.goal_threshold
    
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
        '''
        dirn = np.array(randvex) - np.array(nearvex)
        length = np.linalg.norm(dirn)
        dirn = (dirn / length) * min (stepSize, length)

        newvex = (nearvex[0]+dirn[0], nearvex[1]+dirn[1])
        return newvex
        '''
        
        dirn = np.array([sampled_node_x, sampled_node_y]) - np.array([nearest_node_x, nearest_node_y])
        length = np.linalg.norm(dirn)
        dirn = (dirn / length) * min (expansion_distance, length)
        
        sampled_node_x = nearest_node_x + dirn[0]
        sampled_node_y = nearest_node_y + dirn[1]
        
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
        
    def plan(self, waypoints, occupancy_grid, pose, start_pos=(0.5,0.)):
        '''
        Plan a path through the waypoints given the occupancy grid.
        
        Args:
        - waypoints: ndarray (n, 5) waypoints
        - occupancy_grid: Occupancies object
        - pose: Pose object, current pose in global frame
        - start_pos: starting position in the ego frame, default is (0,0)
        '''
        

        self.waypoints = waypoints
        self.occupancy_grid = occupancy_grid
        lookahead = self.occupancy_grid.lookahead_distance
        
        candidate_goal_point = np.array([pose.x + lookahead * np.cos(pose.theta), 
                                         pose.y + lookahead * np.sin(pose.theta)])
        
        # find waypoint closest to the candidate goal point
        target_waypoint_idx = np.argmin(np.linalg.norm(self.waypoints[:, :2] - candidate_goal_point, axis=1))
        self.target_waypoint = waypoints[target_waypoint_idx, :2]
        self.target_waypoint = pose.global_position_to_pose_frame(self.target_waypoint)
        
        self.G = PlanGraph(start_pos)
        self.is_goal_reached = False
        for i in range(self.max_iterations):
            # sample a point
            position = self._sample_free()
            new_node = PlanNode(position[0], position[1])
            
            nearest_node_id = self.G.get_nearest_node_idx(new_node)
            nearest_node = self.G.get_node(nearest_node_id)
            self._steer(new_node, nearest_node)
            
            if not self._check_collision(new_node, nearest_node):
                self.G.add_node(new_node)
                self.G.add_edge(nearest_node_id, new_node.id, parent_id=nearest_node_id, add_cost=True)
                
                # rewire
                near_node_ids = self.G.get_near_node_ids(new_node, self.search_radius)
                for near_node_id in near_node_ids:
                    near_node = self.G.get_node(near_node_id)
                    if not self._check_collision(new_node, near_node):
                        # check if the cost of the new path is less than the cost of the old path
                        new_cost = near_node.cost + np.linalg.norm(new_node.position - near_node.position)
                        
                        if new_cost < near_node.cost:
                            self.G.add_edge(nearest_node_id, near_node_id, parent_id=nearest_node_id, add_cost=True)
            else:
                continue
            
            if self._is_goal_reached(new_node) and i > self.max_iterations // 2:
                self.destination_node_id = new_node.id
                self.is_goal_reached = True
                break
        
        return self.is_goal_reached
            
    def dijkstra(self, start_node_id, goal_node_id):
        '''
        Dijkstra's algorithm.
        
        Args:
        - start_node_id: id of the start node
        - goal_node_id: id of the goal node
        
        Returns:
        - path: list of node ids
        '''
        pass