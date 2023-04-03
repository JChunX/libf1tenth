import numpy as np

from libf1tenth.planning.waypoints import Waypoints
from libf1tenth.planning.frenet import FrenetFrame
from libf1tenth.planning.path_planner import PathPlanner
from libf1tenth.planning.polynomial import QuarticPolynomial, \
                                           QuinticPolynomial
                                           
# cost weights
K_J = 0.1
K_T = 0.1
K_D = 10.0
K_LAT = 1.0
K_LON = 1.0
                                           
class FrenetPath:

    def __init__(self, frenet_frame, t, s, d, s_d, d_d, s_dd, d_dd, s_ddd, d_ddd):
        '''
        FrenetPath is a path in the Frenet frame.
        '''
        self.frenet_frame = frenet_frame
        self.t = t # time
        
        self.d = d # lateral offset
        self.d_d = d_d # lateral velocity
        self.d_dd = d_dd # lateral acceleration
        self.d_ddd = d_ddd # lateral jerk
        
        self.s = s # path progress
        self.s_d = s_d # path velocity
        self.s_dd = s_dd # path acceleration
        self.s_ddd = s_ddd # path jerk
        
        self.cd = 0.0 # cost of lateral offset
        self.cv = 0.0 # cost of path velocity
        self.cf = 0.0 # overall cost
        
        self.x, self.y = self.frenet_frame.frenet_to_cartesian(self.s, self.d)
        
    def is_collision(self, pose, occupancy_grid):
        '''
        Checks if the path is in collision with the occupancy grid.
        
        Args:
        - pose: Pose object
        - occupancy_grid: Occupancies object
        
        Returns:
        - True if the path is in collision, False otherwise
        '''
        
        positions = np.vstack((self.x, self.y))
        positions_local = pose.global_position_to_pose_frame(positions)
            
        is_collision = occupancy_grid.is_collision(
            'laser', positions_local[0,:], positions_local[1,:])
        
        return is_collision
    
    def to_waypoints(self):
        '''
        Returns a Waypoints object of the path.
        '''
        waypoints = Waypoints.from_numpy(np.vstack((self.x,self.y,self.s_d)).T)
        return waypoints
        

class FrenetPlanner(PathPlanner):
    '''
    FrenetPlanner plans a path using the Frenet frame while avoiding obstacles.
    '''
    def __init__(self, waypoints, left_lim=0.8, right_lim=0.8, lane_width=0.2, 
                                  t_min=4.0, t_max=4.5, t_step=0.25, dt=0.01,
                                  num_v_steps=1, v_step=0.5):
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
        - num_v_steps: number of velocity steps to consider
        - v_step: velocity step size
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
        
        self.num_v_steps = num_v_steps
        self.v_step = v_step
        
        self.max_speed = 15.0
        self.max_accel = 10.0
        
        self.current_path = None
        
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
            
        # replan condition
        if (self.current_path is None
                or self.frenet_frame.progress_diff(self.current_path.s[0], s0) > 5.0
                or self.current_path.is_collision(pose, occupancy_grid)):
            
            frenet_paths, costs = self._generate_frenet_paths(s0, s0_dot, s0_ddot, d0, 0, 0)
            valid_indices = self._check_valid(frenet_paths, pose, occupancy_grid)

            valid_paths = frenet_paths[valid_indices]
            valid_costs = costs[valid_indices]
            
            if len(valid_paths) > 0:
                best_path = valid_paths[np.argmin(valid_costs)]
                self.current_path = best_path
                
        if self.current_path is None:
            return None, False

        return self.current_path.to_waypoints(), True
    
    def _generate_frenet_paths(self, s0, s0_dot, s0_ddot, d0, d0_dot, d0_ddot):
        
        frenet_paths = []
        costs = []
        
        for d_target in np.arange(-self.right_lim, self.left_lim+0.01, self.lane_width):
            for t_target in np.arange(self.t_min, self.t_max, self.t_step):
                lateral_poly = QuinticPolynomial(d0, d0_dot, d0_ddot, d_target, 0.0, 0.0, t_target)
    
                t = np.arange(0, t_target, self.dt)
                d = lateral_poly.calc_point(t)
                d_d = lateral_poly.calc_first_derivative(t)
                d_dd = lateral_poly.calc_second_derivative(t)
                d_ddd = lateral_poly.calc_third_derivative(t)
                
                v_target = max(s0_dot, 1.0)
                longitudinal_poly = QuarticPolynomial(s0, s0_dot, s0_ddot, v_target, 0.0, t_target)
                
                s = longitudinal_poly.calc_point(t)
                s_d = longitudinal_poly.calc_first_derivative(t)
                s_dd = longitudinal_poly.calc_second_derivative(t)
                s_ddd = longitudinal_poly.calc_third_derivative(t)
                
                frenet_path = FrenetPath(self.frenet_frame, t, s, d, s_d, d_d, s_dd, d_dd, s_ddd, d_ddd)
                
                # square of jerk
                Jd = sum(np.power(frenet_path.d_ddd, 2))  
                Js = sum(np.power(frenet_path.s_ddd, 2))
                
                # square of diff from desired velocity
                ds = (s0_dot - frenet_path.s_d[-1])**2
                
                frenet_path.cd = K_J * Jd + K_T * t_target + K_D * frenet_path.d[-1]**2
                frenet_path.cv = K_J * Js + K_T * t_target + K_D * ds
                frenet_path.cf = K_LAT * frenet_path.cd + K_LON * frenet_path.cv
                
                frenet_paths.append(frenet_path)
                costs.append(frenet_path.cf)
                    
        return np.array(frenet_paths), np.array(costs)
    
    def _check_valid(self, frenet_paths, pose, occupancy_grid):
        valid_indices = []
        for i, frenet_path in enumerate(frenet_paths):
            if not self._is_valid(frenet_path):
                continue
            if frenet_path.is_collision(pose, occupancy_grid):
                continue
            else:
                valid_indices.append(i)
                
        return valid_indices

    def _is_valid(self, frenet_path):
        '''
        Checks if the frenet path is within the limits of the frenet frame.
        '''
        is_valid = True
        # speed check
        if np.max(frenet_path.s_d) > self.max_speed:
            is_valid = False
        # acceleration check
        if np.max(np.abs(frenet_path.s_dd)) > self.max_accel:
            is_valid = False

        return is_valid