import numpy as np

from libf1tenth.planning.waypoints import Waypoints
from libf1tenth.planning.frenet import FrenetFrame
from libf1tenth.planning.path_planner import PathPlanner
from libf1tenth.planning.polynomial import QuarticPolynomial, \
                                           QuinticPolynomial
                                           
# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0
                                           
class FrenetPath:

    def __init__(self):
        '''
        FrenetPath is a path in the Frenet frame.
        '''
        self.t = [] # time
        
        self.d = [] # lateral offset
        self.d_d = [] # lateral velocity
        self.d_dd = [] # lateral acceleration
        self.d_ddd = [] # lateral jerk
        
        self.s = [] # path progress
        self.s_d = [] # path velocity
        self.s_dd = [] # path acceleration
        self.s_ddd = [] # path jerk
        
        self.cd = 0.0 # cost of lateral offset
        self.cv = 0.0 # cost of path velocity
        self.cf = 0.0 # overall cost

        self.x = [] # x coordinate
        self.y = [] # y coordinate
        self.yaw = [] # yaw angle
        self.ds = [] # distance between points
        self.c = [] # curvature

class FrenetPlanner(PathPlanner):
    '''
    FrenetPlanner plans a path using the Frenet frame while avoiding obstacles.
    '''
    def __init__(self, waypoints, left_lim=1.0, right_lim=1.0, lane_width=0.25, 
                                  t_min=0.5, t_max=1.5, t_step=0.1,
                                  num_v_steps=5, v_step=0.2):
        '''
        initializes the frenet planner with waypoints and a frenet frame
        waypoints: Waypoints object
        '''
        self.waypoints = waypoints
        self.frenet_frame = FrenetFrame(waypoints)
        
        self.left_lim = left_lim
        self.right_lim = right_lim
        self.lane_width = lane_width
        self.t_min = t_min
        self.t_max = t_max
        self.t_step = t_step
        
        self.num_v_steps = num_v_steps
        self.v_step = v_step
        
    def plan(self, occupancy_grid, pose):
        '''
        Plan a path through the waypoints given the occupancy grid.
        
        Args:
        - occupancy_grid: Occupancies object
        - pose: Pose object, current pose of the vehicle in global frame
        '''
        x, y, s_dot, yaw = pose.x, pose.y, pose.velocity, pose.yaw
        s, d = self.frenet_frame.cartesian_to_frenet(x, y)
        s_ddot = 0.0 ####### hack #######
        frenet_paths = self._generate_frenet_paths(s, s_dot, s_ddot, d, 0, 0)
        valid_paths = self._generate_valid_global_paths(frenet_paths, pose, occupancy_grid)
        
        min_cost = np.inf
        best_path = None
        for path in valid_paths:
            cost = path.cf
            if cost < min_cost:
                min_cost = cost
                best_path = path

        return best_path
    
    def _generate_frenet_paths(self, s, s_dot, s_ddot, d, d_dot, d_ddot):
        
        frenet_paths = []
        
        for d_target in np.arange(-self.left_lim, self.right_lim, self.lane_width):
            for t_target in np.arange(self.t_min, self.t_max, self.t_step):
                lateral_poly = QuarticPolynomial(d, d_dot, d_ddot, d_target, 0, 0, t_target)
    
                t = np.arange(0, t_target, self.t_step)
                d = lateral_poly.calc_point(frenet_path.t)
                d_d = lateral_poly.calc_first_derivative(frenet_path.t)
                d_dd = lateral_poly.calc_second_derivative(frenet_path.t)
                d_ddd = lateral_poly.calc_third_derivative(frenet_path.t)
                
                for v_target in np.arange(s_dot-self.num_v_steps*self.v_step, s_dot+self.num_v_steps*self.v_step, self.v_step):
                    frenet_path = FrenetPath()
                    longitudinal_poly = QuinticPolynomial(s, s_dot, s_ddot, v_target, 0, 0, t_target)
                    
                    frenet_path.t = t
                    frenet_path.d = d
                    frenet_path.d_d = d_d
                    frenet_path.d_dd = d_dd
                    frenet_path.d_ddd = d_ddd
                    
                    frenet_path.s = longitudinal_poly.calc_point(frenet_path.t)
                    frenet_path.s_d = longitudinal_poly.calc_first_derivative(frenet_path.t)
                    frenet_path.s_dd = longitudinal_poly.calc_second_derivative(frenet_path.t)
                    frenet_path.s_ddd = longitudinal_poly.calc_third_derivative(frenet_path.t)
                    
                    # square of jerk
                    Jd = sum(np.power(frenet_path.d_ddd, 2))  
                    Js = sum(np.power(frenet_path.s_ddd, 2))
                    
                    # square of diff from desired velocity
                    ds = (s_dot - frenet_path.s_d[-1])**2
                    
                    frenet_path.cd = K_J * Jd + K_T * t_target + K_D * frenet_path.d[-1]**2
                    frenet_path.cv = K_J * Js + K_T * t_target + K_D * ds
                    frenet_path.cf = K_LAT * frenet_path.cd + K_LON * frenet_path.cv
                    
                    frenet_paths.append(frenet_path)
                    
        return frenet_paths
    
    def _generate_valid_global_paths(self, frenet_paths, pose, occupancy_grid):
        valid_paths = []
        for frenet_path in frenet_paths:
            x,y = self.frenet_frame.frenet_to_cartesian(frenet_path.s, frenet_path.d)
            velocity = frenet_path.s_d
            
            positions = np.vstack((x,y))
            positions_local = pose.global_position_to_pose_frame(positions)
            
            if (occupancy_grid.is_collision(
                'laser', positions_local[0,:], positions_local[1,:])
                or self._check_limits(frenet_path)):
                continue
            
            else: 
                waypoints = Waypoints.from_numpy(np.vstack((x,y,velocity)).T)
                valid_paths.append(waypoints)
                
        return valid_paths

    def _check_limits(self, frenet_path):
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