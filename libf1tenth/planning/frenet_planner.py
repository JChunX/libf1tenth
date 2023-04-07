import time

import numpy as np
from libf1tenth.planning.frenet import FrenetFrame
from libf1tenth.planning.path_planner import PathPlanner
from libf1tenth.planning.polynomial import QuarticPolynomial, QuinticPolynomial
from libf1tenth.planning.waypoints import Waypoints

# cost weights
K_J = 0.1
K_T = -0.1
K_D = 50.0
K_S = -100.0
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
        self.path_length = self.frenet_frame.progress_diff(s[0], s[-1])
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
        '''
        Checks num. of waypoints is in collision with the occupancy grid.
        
        Args:
        - pose: Pose object
        - occupancy_grid: Occupancies object
        
        Returns:
        - num_collisions: number of waypoints in collision
        '''
        
        positions = np.vstack((self.x, self.y))
        positions_local = pose.global_position_to_pose_frame(positions)
            
        _, num_collisions = occupancy_grid.is_collision(
            'laser', positions_local[0,:], positions_local[1,:])
        
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
    def __init__(self, waypoints, left_lim=1.0, right_lim=1.0, lane_width=0.1, 
                                  t_min=2.0, t_max=12.0, t_step=1.0, dt=0.2):
        '''
        Initializes the frenet planner with waypoints and a frenet frame
        
        Args:
        - waypoints: (n, 5) array of waypoints [x, y, velocity, yaw, curvature]
        - left_lim: left limit of the track in meters
        - right_lim: right limit of the track in meters
        - lane_width: width of a lane in meters
        - t_min: minimum time to reach the goal
        - t_max: maximum time to reach the goal
        - t_step: time step for discretizing the goal
        - dt: time step for discretizing the path
        '''
        self.waypoints = waypoints
        self.frenet_frame = FrenetFrame(waypoints)
        
        self.left_lim = left_lim
        self.right_lim = right_lim
        self.lane_width = lane_width
        self.t_min = t_min
        self.t_max = t_max
        self.t_step = t_step
        self.dt = dt
        
        self.max_speed = 15.0
        self.max_accel = 10.0
        
        self.current_path = None
        self.plan_time = -np.inf
        
    def plan(self, occupancy_grid, pose, logger=None):
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
        logger.info('d0_dot: {}'.format(d0_dot))
            
        frenet_paths, costs = self._generate_frenet_paths(s0, s0_dot, s0_ddot, d0, d0_dot, 0, pose, occupancy_grid)
        valid_indices = self._check_valid(frenet_paths)

        valid_paths = frenet_paths[valid_indices]
        valid_costs = costs[valid_indices]
        #valid_costs[valid_indices] -= 9000.0
        
        if len(valid_paths) > 0:
            best_path = valid_paths[np.argmin(valid_costs)]
            if self.current_path is None:
                self.current_path = best_path
                self.plan_time = time.time()
            
            elif (self.frenet_frame.progress_diff(s0, self.current_path.s[-1]) < 2.0
                    or self.current_path.num_collisions(pose, occupancy_grid) > 0):
                self.current_path = best_path
                self.plan_time = time.time()
        else:
            logger.info('No valid paths found')
                
        if self.current_path is None:
            return None, None, False

        return self.current_path.to_waypoints(), [path.to_waypoints() for path in frenet_paths], True
    
    def _generate_frenet_paths(self, s0, s0_dot, s0_ddot, d0, d0_dot, d0_ddot, pose, occupancy_grid):
        
        frenet_paths = []
        costs = []
        
        for d_target in np.arange(-self.right_lim+d0, self.left_lim+d0+0.01, self.lane_width):
            d_target += np.random.uniform(-self.lane_width/2.0, self.lane_width/2.0)
            for t_target in np.arange(self.t_min, self.t_max+0.01, self.t_step):
                lateral_poly = QuinticPolynomial(d0, d0_dot, d0_ddot, d_target, 0.0, 0.0, t_target)
    
                t = np.arange(0, t_target, self.dt)
                d = lateral_poly.calc_point(t)
                
                v_target = max(s0_dot, 1.0)
                longitudinal_poly = QuarticPolynomial(s0, s0_dot, s0_ddot, v_target, 0.0, t_target)
                
                s = longitudinal_poly.calc_point(t)

                
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
                Js = sum(np.power(s_ddd, 2))
                
                cd = K_J * Jd + K_T * t_target + K_D * frenet_path.d[-1]**2
                cv = K_J * Js + K_T * t_target + K_S * self.frenet_frame.progress_diff(s0, frenet_path.s[-1])**2
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