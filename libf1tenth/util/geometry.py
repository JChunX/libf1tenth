import numpy as np
from numba import njit


@njit
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