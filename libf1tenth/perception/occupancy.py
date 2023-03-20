from typing import Union

import numpy as np
from numba import njit
from nav_msgs.msg import OccupancyGrid
from scipy.ndimage import binary_dilation

from libf1tenth.util.geometry import bresenham
from libf1tenth.util.transformations import to_homogenous


class Occupancies:
    '''
    Occupancies
    
    - represents occupancy of the car's environment. 
    
    - contains multiple layers of occupancy at a 
    standard resolution and size. 
    
    - for all layers, the resolution, x/y shape, and x/y origin
    are the same. 
    
    - each layer contains a occupancy map with probability [0-1]
    
    Args:
    - resolution: resolution of the occupancy grid
    - x_size: number of cells in x direction
    - y_size: number of cells in y direction
    '''
    
    def __init__(self, resolution, x_size, y_size):
        self.resolution = resolution
        self.x_size = x_size
        self.y_size = y_size
        self.x_origin = 0.0
        self.y_origin = self.y_size * self.resolution / 2.0 # units: m
        
        self.lookahead_distance = self.resolution * self.x_size
        
        self.layers = {}
        
    @property
    def _pc_to_grid(self):
        '''
        Returns transform from pointcloud x, y to grid x, y
        
        x_grid = x_pc + x_origin
        y_grid = y_pc + y_origin
        '''
        return np.array([[1,0,self.x_origin],
                         [0,1,self.y_origin],
                         [0,0,1]])
        
    @property
    def _grid_to_pc(self):
        '''
        Returns transform from grid x, y to pointcloud x, y
        
        x_pc = x_grid - x_origin
        y_pc = y_grid - y_origin
        '''
        return np.array([[1,0,-self.x_origin],
                         [0,1,-self.y_origin],
                         [0,0,1]])
    
    def pc_to_grid_indices(self, x_pc: Union[float, np.ndarray], y_pc: Union[float, np.ndarray]):
        '''
        Converts pointcloud positions to grid indices
        
        Args:
        - x_pc: ndarray / scalar x positions of points in odom frame
        - y_pc: ndarray / scalar positions of points in odom frame
        
        Returns:
        - x_idx: ndarray of shape (2, ) / scalar representing x indices of points in grid
        - y_idx: ndarray of shape (2, ) / scalar representing y indices of points in grid
        '''
        xy_pc = to_homogenous(x_pc, y_pc)
        xy_idx = ((self._pc_to_grid @ xy_pc).T / self.resolution).astype(int)[:, :2].reshape(-1, 2)
        
        if np.isscalar(x_pc):
            return xy_idx[0,0], xy_idx[0,1]
        
        return xy_idx[:,0], xy_idx[:,1]
    
    def grid_indices_to_pc(self, x_idx, y_idx):
        '''
        Converts grid indices to pointcloud positions
        
        Args:
        - x_idx: ndarray / scalar x indices of points in grid
        - y_idx: ndarray / scalar y indices of points in grid
        
        Returns:
        - x_pc: ndarray of shape (2, ) / scalar representing x positions of points in odom frame
        - y_pc: ndarray of shape (2, ) / scalar representing y positions of points in odom frame
        '''
        x_grid_pos = x_idx * self.resolution
        y_grid_pos = y_idx * self.resolution
        xy_grid_pos = to_homogenous(x_grid_pos, y_grid_pos)
        
        xy_pc = (self._grid_to_pc @ xy_grid_pos).T.astype(float)[:,:2].reshape(-1, 2)
        
        if np.isscalar(x_idx):
            return xy_pc[0,0], xy_pc[0,1]
        
        return xy_pc[:,0], xy_pc[:,1]
    
    def create_layer(self, layer_name):
        self.layers[layer_name] = {'occupancy': np.zeros((self.x_size, 
                                            self.y_size), 
                                            dtype=float)}
        return self
        
    def set_layer_property(self, layer_name, property_name, value):
        '''
        Sets property to the given layer
        '''
        assert value.shape == (self.x_size, self.y_size), "layer shape does not match occupancy shape"
        self.layers[layer_name][property_name] = value
        
    def set_layer_occupancy_from_pc(self, layer_name, x_pc, y_pc, confidence):
        '''
        Sets layer to the given pointcloud in odom frame with confidence between 0 and 1
        
        Args:
        - layer_name: name of the layer to set
        - x_pc: x positions of points in odom frame
        - y_pc: y positions of points in odom frame
        - confidence: confidence of each point in the pointcloud, or a single value
        '''
        # if confidence is a single value, repeat it to match the number of points
        if np.isscalar(confidence):
            confidence = np.repeat(confidence, x_pc.shape[0])
        
        assert np.min(confidence) >= 0 and np.max(confidence) <= 1, "confidence must be between 0 and 1"
        
        # convert pointcloud positions to grid indices
        x_occ_idx, y_occ_idx = self.pc_to_grid_indices(x_pc, y_pc)
        
        # remove points that are out of bounds
        valid_indices = ((x_occ_idx >= 0)
                                & (x_occ_idx < self.x_size)
                                & (y_occ_idx >= 0)
                                & (y_occ_idx < self.y_size))
        x_occ_idx = x_occ_idx[valid_indices]
        y_occ_idx = y_occ_idx[valid_indices]
        confidence = confidence[valid_indices]

        self.layers[layer_name]['occupancy'][x_occ_idx, y_occ_idx] = confidence
        
    def dilate_layer(self, layer_name, iterations=1):
        '''
        dilates the occupancy of given layer by the given number of iterations
        this operation will dilate any values > 0.5 and discard probabilities
        
        Args:
        - layer_name: name of the layer to dilate
        - iterations: number of times to dilate the layer
        '''
        # set all values > 0.5 to 1 and all values <= 0.5 to 0
        occupancy = self.layers[layer_name]['occupancy']
        occupancy = (occupancy > 0.5).astype(float)
        self.layers[layer_name]['occupancy'] = binary_dilation(
                                                    occupancy, 
                                                    iterations=iterations).astype(float)
        
    def is_collision_free(self, layer_name, x_pc, y_pc):
        '''
        Determines if the single given pointcloud location is collision free
        
        Args:
        - layer_name: name of the layer to check
        - x_pc: x position of a single point in odom frame
        - y_pc: y position of a single point in odom frame
        
        Returns:
        - is_collision_free: True if the point is collision free
        '''
        x_idx, y_idx = self.pc_to_grid_indices(x_pc, y_pc)
        is_collision_free = self.layers[layer_name]['occupancy'][x_idx, y_idx] == 0
        return is_collision_free
        
    def check_line_collision(self, layer_name, x1, y1, x2, y2):
        '''
        Checks if the given line segment is collision free
        
        Args:
        - layer_name: name of the layer to check
        - x1: x position of the first point in odom frame
        - y1: y position of the first point in odom frame
        - x2: x position of the second point in odom frame
        - y2: y position of the second point in odom frame
        
        Returns:
        - is_collision_free: True if the line segment is collision free 
        '''
        
        occ_grid = self.layers[layer_name]['occupancy']
        
        start_x_idx, start_y_idx = self.pc_to_grid_indices(x1, y1)
        end_x_idx, end_y_idx = self.pc_to_grid_indices(x2, y2)
        
        # convert line segment to grid indices using Bresenham's algorithm
        x_indices, y_indices = bresenham(start_x_idx, start_y_idx, end_x_idx, end_y_idx)
        # check if any of the line segment points are occupied
        is_collision_free = np.all(occ_grid[x_indices, y_indices] == 0)
        return is_collision_free
        
    def to_msg(self, layer_name, frame_id):
        '''
        Returns occupancy grid message for the given layer
        
        Args:
        - layer_name: name of the layer to convert to a message
        - pose: pose of the car wrt map frame
        - frame_id: frame id of the occupancy grid message
        
        Returns: 
        - msg: OccupancyGrid message
        '''
        # convert to 8-bit occupancy grid with confidence values rescaled between 0-100
        occ_grid_vals = (self.layers[layer_name]['occupancy'] * 100).astype(np.int8)
        occ_grid_vals = np.clip(occ_grid_vals, 0, 100)
        occ_grid_vals = occ_grid_vals.flatten(order='F').tolist()

        msg = OccupancyGrid()
        msg.header.frame_id = frame_id
        msg.info.resolution = self.resolution
        msg.info.width = self.x_size
        msg.info.height = self.y_size

        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = -self.y_origin
        msg.data = occ_grid_vals
        
        return msg
    
    def to_image(self, layer_name):
        occ_grid = self.layers[layer_name]['occupancy']
        # reverse the row
        occ_grid = np.flip(occ_grid, axis=0)
        # reverse the column
        occ_grid = np.flip(occ_grid, axis=1)
        # rescale the values between 0-255
        occ_grid = (occ_grid * 255).astype(np.uint8)
        
        return occ_grid
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # init test occupancy grid
    occ = Occupancies(0.04, 120, 80)
    occ.create_layer('test').set_layer_property('test', 'occupancy', np.zeros((120, 80)))
    occ.set_layer_occupancy_from_pc('test', x_pc=np.array([2]), 
                                        y_pc=np.array([0]), 
                                        confidence=1.0)
    
    # draw an obstacle
    start_x_idx = 0
    start_y_idx = 0
    end_x_idx = 30
    end_y_idx = 80-1
    x_indices, y_indices = bresenham(start_x_idx, start_y_idx, end_x_idx, end_y_idx)
    occ.layers['test']['occupancy'][x_indices, y_indices] = 1
    
    # create a line that does not intersect the obstacle
    x1 = 0
    y1 = 0
    x2 = 0.5
    y2 = 0
    assert occ.check_line_collision('test', x1, y1, x2, y2) == True, 'Intersection test failed: expected True'
    
    # create a line that does intersect the obstacle
    x1 = 0
    y1 = 1.75
    x2 = 2.0
    y2 = 1.75
    assert occ.check_line_collision('test', x1, y1, x2, y2) == False, 'Intersection test failed: expected False'
    
    # draw the line
    start_x_idx, start_y_idx = occ.pc_to_grid_indices(x1, y1)
    end_x_idx, end_y_idx = occ.pc_to_grid_indices(x2, y2)

    x_indices, y_indices = bresenham(start_x_idx, start_y_idx, end_x_idx, end_y_idx)
    occ.layers['test']['occupancy'][x_indices, y_indices] = 1
    
    img = occ.to_image('test')
    plt.imshow(img)
    plt.show()
    
    