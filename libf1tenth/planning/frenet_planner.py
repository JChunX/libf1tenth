import math
import time

import numpy as np

from libf1tenth.planning.frenet import FrenetFrame
from libf1tenth.planning.graph import FrenetPlanGraph, PlanNode
from libf1tenth.planning.path_planner import PathPlanner
from libf1tenth.planning.rrt import RRTPlanner
from libf1tenth.planning.polynomial import QuarticPolynomial, QuinticPolynomial
from libf1tenth.planning.waypoints import Waypoints

# cost weights
K_J = 0.001 # jerk
K_T = -0.1 # time
K_D = 10.0 # lateral offset
K_S = -0.1 # path progress
K_LAT = 1.0
K_LON = 1.0
                                           
class FrenetPath:

    def __init__(self, frenet_frame, t, s, d):
        '''
        FrenetPath is a path in the Frenet frame.
        '''
        self.frenet_frame = frenet_frame
        self.t = t # time
        self.s = s # path progress
        self.path_length = self.frenet_frame.wrapped_diff(self.s[0], self.s[-1])
        self.d = d # lateral offset
        
        self.d_d = [] # lateral velocity
        self.d_dd = [] # lateral acceleration
        self.d_ddd = [] # lateral jerk
        
        self.s_d = [] # path velocity
        self.s_dd = [] # path acceleration
        self.s_ddd = [] # path jerk
        
        self.cd = 0.0 # cost of lateral offset
        self.cv = 0.0 # cost of path velocity
        self.cf = 0.0 # overall cost
        
        self.x, self.y = self.frenet_frame.frenet_to_cartesian(self.s, self.d)
        
    def finalize(self, d_d, d_dd, d_ddd, s_d, s_dd, s_ddd, cd, cv, cf):
        '''
        Finalizes the path by setting the derivatives and computing the costs.
        '''
        self.d_d = d_d
        self.d_dd = d_dd
        self.d_ddd = d_ddd
        self.s_d = s_d
        self.s_dd = s_dd
        self.s_ddd = s_ddd
        self.cd = cd
        self.cv = cv
        self.cf = cf
        
    def num_collisions(self, pose, occupancy_grid):
        num_collisions, _ = Waypoints.check_collisions(self.x, self.y, pose, occupancy_grid)
        return num_collisions
    
    def to_waypoints(self):
        '''
        Returns a Waypoints object of the path.
        '''
        waypoints = Waypoints.from_numpy(np.vstack((self.x,self.y,self.s_d)).T).upsample(5)
        return waypoints
        

class FrenetPlanner(PathPlanner):
    '''
    FrenetPlanner plans a path using the Frenet frame while avoiding obstacles.
    '''
    def __init__(self, waypoints, left_lim=0.4, right_lim=0.4, lane_width=0.4, t_step=0.1, dt=0.2, logger=None):
        '''
        Initializes the frenet planner with waypoints and a frenet frame
        
        Args:
        - waypoints: (n, 5) array of waypoints [x, y, velocity, yaw, curvature]
        - left_lim: left limit of the track in meters
        - right_lim: right limit of the track in meters
        - lane_width: width of a lane in meters
        - dt: time step for discretizing the path
        '''
        super().__init__()
        self.waypoints = waypoints
        self.frenet_frame = FrenetFrame(waypoints)
        
        self.left_lim = left_lim
        self.right_lim = right_lim
        self.lane_width = lane_width

        self.dt = dt
        
        self.max_speed = 15.0
        self.max_accel = 10.0
        
        self.current_path = None
        self.plan_time = -np.inf
        
        self.logger = logger
        
    def plan(self, occupancy_grid, pose):
        '''
        Plan a path through the waypoints given the occupancy grid.
        
        Args:
        - occupancy_grid: Occupancies object
        - pose: Pose object, current pose of the vehicle in global frame
        '''
        x0, y0, s0_dot = pose.x, pose.y, pose.velocity
        s0, d0 = self.frenet_frame.cartesian_to_frenet(x0, y0)
        s0_ddot = 0.0 ####### hack #######
        d0_dot = np.sin(pose.theta - self.frenet_frame.cs_syaw(s0)) * s0_dot
            
        frenet_paths, costs = self._generate_frenet_paths(s0, s0_dot, s0_ddot, d0, d0_dot, 0, pose, occupancy_grid)
        valid_indices = self._check_valid(frenet_paths)

        valid_paths = frenet_paths[valid_indices]
        valid_costs = costs[valid_indices]
        #valid_costs[valid_indices] -= 9000.0
        
        if len(valid_paths) > 0:
            path_dt = time.time() - self.plan_time
            best_path = valid_paths[np.argmin(valid_costs)]
            if self.current_path is None:
                self.current_path = best_path
                self.plan_time = time.time()
            
            elif (path_dt > 0.2
                    or self.current_path.num_collisions(pose, occupancy_grid) > 0):
                self.current_path = best_path
                self.plan_time = time.time()
        #else:
            #self.logger.info('No valid paths found')
                
        if self.current_path is None:
            return None, None, False

        return self.current_path.to_waypoints(), [path.to_waypoints() for path in frenet_paths], True
    
    def _augment_t(self, s0_dot):
        # augment t by s0_dot
        if s0_dot < 1.0:
            t_plan = 3.0
        else:
            t_plan = 4.0 / s0_dot
            
        return t_plan
    
    def _generate_frenet_paths(self, s0, s0_dot, s0_ddot, d0, d0_dot, d0_ddot, pose, occupancy_grid):
        
        frenet_paths = []
        costs = []
        
        t_plan = self._augment_t(s0_dot)
        
        t_min = t_plan
        t_max = t_plan
        t_step = t_plan / 50.0
        
        for d_target in [-self.right_lim, self.left_lim]:#np.arange(-self.right_lim, self.left_lim+0.01, self.lane_width):
            #d_target += np.random.uniform(-self.lane_width/2.0, self.lane_width/2.0)

            for t_target in np.arange(t_min, t_max+0.01, t_step):
                lateral_poly = QuinticPolynomial(d0, d0_dot, d0_ddot, d_target, 0.0, 0.0, t_target)
    
                t = np.arange(0, t_target, self.dt)
                d = lateral_poly.calc_point(t)
                
                v_target = max(s0_dot, 1.0)
                longitudinal_poly = QuarticPolynomial(s0, s0_dot, s0_ddot, v_target, 0.0, t_target)
                
                s = self.frenet_frame.wrap(longitudinal_poly.calc_point(t))

                frenet_path = FrenetPath(self.frenet_frame, t, s, d)
                
                if frenet_path.num_collisions(pose, occupancy_grid) > 0:
                    continue
                
                s_d = longitudinal_poly.calc_first_derivative(t)
                s_dd = longitudinal_poly.calc_second_derivative(t)
                s_ddd = longitudinal_poly.calc_third_derivative(t)
                d_d = lateral_poly.calc_first_derivative(t)
                d_dd = lateral_poly.calc_second_derivative(t)
                d_ddd = lateral_poly.calc_third_derivative(t)
                
                # square of jerk
                Jd = sum(np.power(d_ddd, 2))  
                #Js = sum(np.power(s_ddd, 2))
                
                cd = K_J * Jd + K_T * t_target + K_D * np.sum(frenet_path.d**2)
                cv = K_T * t_target + K_S * self.frenet_frame.wrapped_diff(s0, frenet_path.s[-1])**2
                cf = K_LAT * cd + K_LON * cv
                
                frenet_path.finalize(d_d, d_dd, d_ddd, s_d, s_dd, s_ddd, cd, cv, cf)
                
                frenet_paths.append(frenet_path)
                costs.append(frenet_path.cf)
                    
        return np.array(frenet_paths), np.array(costs)
    
    def _check_valid(self, frenet_paths):
        valid_indices = []
        for i, frenet_path in enumerate(frenet_paths):
            if not self._constraints_met(frenet_path):
                continue
            else:
                valid_indices.append(i)
                
        return valid_indices

    def _constraints_met(self, frenet_path):
        '''
        Checks if the frenet path is within the limits of the frenet frame.
        '''
        # speed check
        if np.max(frenet_path.s_d) > self.max_speed:
            return False
        # acceleration check
        if np.max(np.abs(frenet_path.s_dd)) > self.max_accel:
            return False

        return True
    
class FrenetRRTStarPlanner(RRTPlanner):
    '''
    RRT*, but in frenet frame.
    '''
    def __init__(self, waypoints, 
                 search_radius,  
                 max_iterations, 
                 expansion_distance, 
                 goal_threshold, 
                 occ_grid_layer='laser', 
                 left_lim=0.5, right_lim=0.5,
                 logger=None):
        
        super().__init__(search_radius, 
                         max_iterations, 
                         expansion_distance, 
                         goal_threshold, 
                         occ_grid_layer,
                         logger)
        
        self.waypoints = waypoints
        self.frenet_frame = FrenetFrame(waypoints)
        
        self.current_path = None
        self.plan_time = -np.inf
        self.lookahead = None
        self.left_lim = left_lim
        self.right_lim = right_lim
        
    def plan(self, occupancy_grid, pose):
        self.occupancy_grid = occupancy_grid
        self.pose = pose
        self.lookahead = self.occupancy_grid.lookahead_distance // 2
        x0, y0, s0dot = self.pose.x, self.pose.y, self.pose.velocity
        s0, d0 = self.frenet_frame.cartesian_to_frenet(x0, y0)
        d0_dot = np.sin(self.pose.theta - self.frenet_frame.cs_syaw(s0)) * s0dot
        sf = s0 + self.lookahead
        self.nominal_s_path = np.linspace(s0, sf, int(self.lookahead*20))
        self.nominal_k_path_norm = self.frenet_frame.cs_sk(self.nominal_s_path)
        self.nominal_k_path_norm = self.nominal_k_path_norm / np.sum(self.nominal_k_path_norm)
        
        self.target_waypoint = np.array([sf, d0])
        self.G = FrenetPlanGraph(self.frenet_frame, (s0, d0))
        self.is_goal_reached = False
        positions = None
        
        for i in range(self.max_iterations):
            position = self._sample_free()
            if positions is None:
                positions = position
            else:
                positions = np.vstack((positions, position))
            new_node = PlanNode(position[0], position[1])
            nearest_node_id = self.G.get_nearest_node_idx(new_node)
            nearest_node = self.G.get_node(nearest_node_id)
            #self._steer(new_node, nearest_node)
            
            if not self._check_collision(new_node, nearest_node):
                self.G.add_node(new_node)
                self.G.add_edge(nearest_node_id, new_node.id, 
                                parent_id=nearest_node_id, cost=self._cost(parent_node=nearest_node, 
                                                                           child_node=new_node))
                
                near_node_ids = self.G.get_near_node_ids(new_node, self.search_radius)

                for near_node_id in near_node_ids:
                    near_node = self.G.get_node(near_node_id)
                    if not self._check_collision(new_node, near_node):
                        new_cost = self._cost(parent_node=new_node, 
                                              child_node=near_node)
                        
                        if new_cost < near_node.cost:
                            self.G.add_edge(new_node.id, near_node_id, parent_id=new_node.id, cost=new_cost)
            else:
                continue

            if self._is_goal_reached(new_node) and i > self.max_iterations//2:
                self.destination_node_id = new_node.id
                self.is_goal_reached = True
                break
        print('debug')
        return self.is_goal_reached
    
    def _cost(self, parent_node, child_node):        
        return parent_node.cost + self.frenet_frame.frenet_distance(
            parent_node.position, child_node.position)
            
    def get_rrt_waypoints(self, velocity=0.5):
        '''
        Get the planned waypoints in the global frame.
        '''
        if not self.is_goal_reached:
            return None
        destination_node = self.G.nodes[self.destination_node_id]
        s_d, d_d = destination_node.position
        x_d, y_d = self.frenet_frame.frenet_to_cartesian(s_d, d_d)
        positions_to_track = np.array([x_d, y_d]).reshape(1, 2)
        
        cur_node = destination_node
        while cur_node.parent is not None:
            cur_node = self.G.nodes[cur_node.parent.id]
            x_cur, y_cur = self.frenet_frame.frenet_to_cartesian(cur_node.position[0], 
                                                                 cur_node.position[1])
            pos = np.array([x_cur, y_cur]).reshape(1, 2)
            positions_to_track = np.vstack((pos, positions_to_track)) # shape (n, 2)
        
        if len(positions_to_track.shape) == 1:
            return None
        
        waypoints_to_track = np.hstack((positions_to_track, 
                                        velocity * np.ones((positions_to_track.shape[0], 1))))
        
        waypoints = Waypoints.from_numpy(waypoints_to_track)#.upsample(10)#.smooth(1)
        
        return waypoints
            
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
        s1, d1 = sampled_node.position
        s2, d2 = nearest_node.position
        x1, y1 = self.frenet_frame.frenet_to_cartesian(s1, d1)
        x2, y2 = self.frenet_frame.frenet_to_cartesian(s2, d2)
        
        local_pos = np.array([[x1,x2],
                              [y1,y2]])
        global_pos = self.pose.global_position_to_pose_frame(local_pos)
        x1, x2 = global_pos[0,0], global_pos[0,1]
        y1, y2 = global_pos[1,0], global_pos[1,1]
        
        is_collision = self.occupancy_grid.check_line_collision(self.occ_grid_layer, x1, y1, x2, y2)
        
        return is_collision
            
    def _sample_free(self):
        '''
        Samples points along frenet trajectory with lookahead
        Weighs sampling probability by curvature and centerline distance
        
        Returns:
        - np.array of shape (2,): sampled point in frenet frame
        '''
        sf, df = self.target_waypoint
        # sample d from normal distribution
        d_sigma = self.left_lim / 3.5
        d_sample = np.random.normal(0.0, d_sigma)
        d_sample = max(min(d_sample, self.left_lim), -self.right_lim)
        # sample s with probability proportional to waypoint curvature
        # prob = self.nominal_k_path_norm + np.max(self.nominal_k_path_norm)
        # prob = prob / np.sum(prob)
        s_sample = np.random.choice(self.nominal_s_path)
        
        return np.array([s_sample, d_sample])
    
    def _is_goal_reached(self, node):
        '''
        Check if the goal is reached.
        
        Args:
        - node (PlanNode): node to check
        
        Returns:
        - is_goal_reached (boolean)
        '''
        goal_s, goal_d = self.target_waypoint
        dist = np.sqrt((self.frenet_frame.wrapped_diff(node.position[0], goal_s))**2 +
                       (node.position[1] - goal_d)**2)

        return dist < self.goal_threshold

    