import math
import time

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def bresenham(start_x_idx, start_y_idx, end_x_idx, end_y_idx):
    '''
    Bresenham's algorithm for finding all points on a line segment
    
    Args:
    - start_x_idx: x index of the start point
    - start_y_idx: y index of the start point
    - end_x_idx: x index of the end point
    - end_y_idx: y index of the end point
    
    Returns:
    - x_indices: x indices of all points on the line segment
    - y_indices: y indices of all points on the line segment
    '''
    dx = abs(end_x_idx - start_x_idx)
    dy = abs(end_y_idx - start_y_idx)
    steep = abs(dy) > abs(dx)
    if steep:
        start_x_idx, start_y_idx = start_y_idx, start_x_idx
        end_x_idx, end_y_idx = end_y_idx, end_x_idx

    # Swap start and end points if necessary and store swap state
    swapped = False
    if start_x_idx > end_x_idx:
        start_x_idx, end_x_idx = end_x_idx, start_x_idx
        start_y_idx, end_y_idx = end_y_idx, start_y_idx
        swapped = True

    # Recalculate differentials
    dx = end_x_idx - start_x_idx
    dy = abs(end_y_idx - start_y_idx)

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if start_y_idx < end_y_idx else -1

    # Iterate over bounding box generating points between start and end
    y = start_y_idx
    points = []
    for x in range(start_x_idx, end_x_idx + 1):
        coord = (y, x) if steep else (x, y)
        points.append(coord)
        error -= dy
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    x_indices = np.array([p[0] for p in points])
    y_indices = np.array([p[1] for p in points])
    return x_indices, y_indices

@njit(cache=True, fastmath=True)
def nearest_point(x, y, waypoints):
    """
    Find the nearest point on the waypoints to the point x, y.
    
    Args:
    - x: x coordinate of the point
    - y: y coordinate of the point
    - waypoints: (num_waypoints, m) ndarray waypoints to search
    
    Output:
    - idx: index of the nearest waypoint
    """
    min_dist = math.inf
    for i in range(waypoints.shape[0]):
        dist = math.sqrt((waypoints[i, 0] - x)**2 + (waypoints[i, 1] - y)**2)
        if dist < min_dist:
            min_dist = dist
            idx = i
    
    return idx

@njit(cache=True, fastmath=True)
def l2_norm(x, y):
    """
    Calculate the L2 norm of n points.
    
    Args:
    - x (ndarray): x coordinates of the points
    - y (ndarray): y coordinates of the points
    
    Returns:
    - norm (ndarray): L2 norm of the points
    """
    return np.sqrt(x**2 + y**2)

@njit(cache=True, fastmath=True)
def argmin(x):
    """
    Find the index of the minimum value in an array.
    
    Args:
    - x (ndarray): array to search
    
    Returns:
    - idx (int): index of the minimum value
    """
    return np.argmin(x)
    
@njit(cache=True, fastmath=True)
def solve_lqr(A, B, Q, R, tolerance, max_num_iteration):
    """
    Iteratively calculating feedback matrix K

    Args:
        A: matrix_a
        B: matrix_b
        Q: matrix_q
        R: matrix_r_
        tolerance: lqr_eps
        max_num_iteration: max_iteration

    Returns:
        K: feedback matrix
    """

    M = np.zeros((Q.shape[0], R.shape[1]))

    AT = A.T
    BT = B.T
    MT = M.T

    P = Q
    num_iteration = 0
    diff = math.inf

    while num_iteration < max_num_iteration and diff > tolerance:
        num_iteration += 1
        P_next = AT @ P @ A - (AT @ P @ B + M) @ \
                 np.linalg.pinv(R + BT @ P @ B) @ (BT @ P @ A + MT) + Q

        # check the difference between P and P_next
        diff = np.abs(np.max(P_next - P))
        P = P_next

    K = np.linalg.pinv(BT @ P @ B + R) @ (BT @ P @ A + MT)

    return K

params = {
    "mu": 1.0489,
    "C_Sf": 4.718,
    "C_Sr": 5.4562,
    "lf": 0.15875,
    "lr": 0.17145,
    "h": 0.074,
    "m": 3.74,
    "I_z": 0.04712,
    "s_min": -0.4189,
    "s_max": 0.4189,
    "sv_min": -0.6,
    "sv_max": 0.6,
    "v_switch": 7.319,
    "a_max": 9.51,
    "v_min": -5.0,
    "v_max": 20.0,
    "width": 0.31,
    "length": 0.58
}

mu = params["mu"]
C_Sf = params["C_Sf"]
C_Sr = params["C_Sr"]
lf = params["lf"]
lr = params["lr"]
h = params["h"]
m = params["m"]
I_z = params["I_z"]
s_min = params["s_min"]
s_max = params["s_max"]
sv_min = params["sv_min"]
sv_max = params["sv_max"]
v_switch = params["v_switch"]
a_max = params["a_max"]
v_min = params["v_min"]
v_max = params["v_max"]
width = params["width"]
length = params["length"]

sig_1 = 2.0 * (C_Sf + C_Sr)
sig_2 = -2.0 * (C_Sf * lf - C_Sr * lr)
sig_3 = -2.0 * (C_Sf * lf**2 + C_Sr * lr**2)

@njit(cache=True, fastmath=True)
def linearized_discrete_lateral_dynamics(velocity, state_size, timestep, wheelbase):
    '''
    Calculate A and b matrices of linearized, discrete system.

    Args:
    - velocity: current vehicle velocity
    - state_size: state size
    - timestep: time step
    - wheelbase: wheelbase

    Returns:
    - Ad: time discrete A matrix
    - Bd: time discrete b matrix
    '''

    #Current vehicle velocity

    #Initialization of the time discrete A matrix
    Ad = np.zeros((state_size, state_size))
    
    if velocity < 1.0:
    
        # A kinematic matrix:
        # [1, dt, 0, 0],
        # [0, 0,  v, 0],
        # [0, 0,  1, dt],
        # [0, 0,  0, 0]
        Ad[0][0] = 1.0
        Ad[0][1] = timestep
        Ad[1][2] = velocity
        Ad[2][2] = 1.0
        Ad[2][3] = timestep
        
    else:
        # https://shbing.github.io/papers/first-author/J7_Design-Analysis-and-Experiments-of-Preview-Path-Tracking-Control-for-Autonomous-Vehicles.pdf
        A = np.array([[0.0, 1.0, 0.0, 0.0],
                    [0.0, -sig_1/(m*velocity), sig_1/m, sig_2/(m*velocity)],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, sig_2/(I_z*velocity), -sig_2/I_z, sig_3/(I_z*velocity)]
                ])
        
        Ad = np.eye(4) + timestep * A

    # b matrix:
    # [0.0, 
    #  0.0, 
    #  0.0, 
    #  v / L]
    Bd = np.zeros((state_size, 1))  # time discrete b matrix
    Bd[3][0] = velocity / wheelbase

    return Ad, Bd

################################################################
# baselines for comparison
def nearest_point_baseline1(x, y, waypoints):
    """
    Find the nearest point on the waypoints to the point x, y.
    
    Args:
    - x: x coordinate of the point
    - y: y coordinate of the point
    - waypoints: (num_waypoints, m) ndarray waypoints to search
    
    Output:
    - idx: index of the nearest waypoint
    """
    dists = np.sqrt((waypoints[:, 0] - x)**2 + (waypoints[:, 1] - y)**2)
    idx = np.argmin(dists)
    return idx

def nearest_point_baseline2(x, y, waypoints):
    """
    Find the nearest point on the waypoints to the point x, y.
    
    Args:
    - x: x coordinate of the point
    - y: y coordinate of the point
    - waypoints: (num_waypoints, m) ndarray waypoints to search
    
    Output:
    - idx: index of the nearest waypoint
    """
    idx = np.argmin(np.linalg.norm(waypoints[:, :2] - np.array([x, y]), axis=1))
    
def l2_norm_baseline(x, y):
    """
    Calculate the L2 norm of n points.
    
    Args:
    - x (ndarray): x coordinates of the points
    - y (ndarray): y coordinates of the points
    
    Returns:
    - norm (ndarray): L2 norm of the points
    """
    norm = np.linalg.norm(np.stack([x, y], axis=1), axis=1)
    return norm

def l2_norm_baseline2(x, y):
    """
    Calculate the L2 norm of n points.
    
    Args:
    - x (ndarray): x coordinates of the points
    - y (ndarray): y coordinates of the points
    
    Returns:
    - norm (ndarray): L2 norm of the points
    """
    norm = np.sqrt(x**2 + y**2)
    return norm

def argmin_baseline(x):
    return np.argmin(x)
    
def solve_lqr_baseline(A, B, Q, R, tolerance, max_num_iteration):
    """
    Iteratively calculating feedback matrix K

    Args:
        A: matrix_a
        B: matrix_b
        Q: matrix_q
        R: matrix_r_
        tolerance: lqr_eps
        max_num_iteration: max_iteration

    Returns:
        K: feedback matrix
    """

    M = np.zeros((Q.shape[0], R.shape[1]))

    AT = A.T
    BT = B.T
    MT = M.T

    P = Q
    num_iteration = 0
    diff = math.inf

    while num_iteration < max_num_iteration and diff > tolerance:
        num_iteration += 1
        P_next = AT @ P @ A - (AT @ P @ B + M) @ \
                 np.linalg.pinv(R + BT @ P @ B) @ (BT @ P @ A + MT) + Q

        # check the difference between P and P_next
        diff = np.abs(np.max(P_next - P))
        P = P_next

    K = np.linalg.pinv(BT @ P @ B + R) @ (BT @ P @ A + MT)

    return K

    
if __name__ == "__main__":
    bench_nearest_pt = False
    bench_lqr = False
    bench_l2_norm = False
    bench_argmin = False
    
    if bench_nearest_pt:
        # nearest point benchmark
        waypoints = np.random.rand(10000, 3)
        x = 0.5
        y = 0.5
        
        start = time.time()
        for i in range(100000):
            nearest_point(x, y, waypoints)
        end = time.time()
        print("nearest_pt_numba: ", end - start)
        
        start = time.time()
        for i in range(100000):
            nearest_point_baseline1(x, y, waypoints)
        end = time.time()
        print("nearest_pt_nonumba: ", end - start)
        
        # start = time.time()
        # for i in range(100000):
        #     nearest_point_baseline2(x, y, waypoints)
        # end = time.time()
        # print("nearest_pt_nonumba2: ", end - start)
    
    if bench_lqr:
        # lqr benchmark
        A = np.random.rand(10, 10)
        B = np.random.rand(10, 2)
        Q = np.random.rand(10, 10)
        R = np.random.rand(2, 2)
        tolerance = 1e-3
        max_num_iteration = 100
        
        start = time.time()
        for i in range(10000):
            solve_lqr(A, B, Q, R, tolerance, max_num_iteration)
        end = time.time()
        print("lqr_numba: ", end - start)
        
        start = time.time()
        for i in range(10000):
            solve_lqr_baseline(A, B, Q, R, tolerance, max_num_iteration)
        end = time.time()
        print("lqr_nonumba: ", end - start)
        
    if bench_l2_norm:
        # l2 norm benchmark
        x = np.random.rand(10000)
        y = np.random.rand(10000)
        
        start = time.time()
        for i in range(100000):
            d = l2_norm(x, y)
        end = time.time()
        print("l2_norm_numba: ", end - start)
        
        start = time.time()
        for i in range(100000):
            d = l2_norm_baseline(x, y)
        end = time.time()
        print("l2_norm_nonumba: ", end - start)
        
        start = time.time()
        for i in range(100000):
            d = l2_norm_baseline2(x, y)
        end = time.time()
        print("l2_norm_nonumba2: ", end - start)
        
    if bench_argmin:
        # argmin benchmark
        x = np.random.rand(10000)
        
        start = time.time()
        for i in range(100000):
            d = argmin(x)
        end = time.time()
        print("argmin_numba: ", end - start)
        
        start = time.time()
        for i in range(100000):
            d = argmin_baseline(x)
        end = time.time()
        print("argmin_nonumba: ", end - start)
