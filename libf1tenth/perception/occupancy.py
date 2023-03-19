import numpy as np
from nav_msgs.msg import OccupancyGrid
from scipy.ndimage import binary_dilation

from libf1tenth.util.transformations import to_homogenous


class Occupancies:
    '''
    OccupancyGrid
    
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
        
    def _pc_to_grid_indices(self, x_pc, y_pc):
        xy_pc = to_homogenous(x_pc, y_pc)
        xy_idx = ((self._pc_to_grid @ xy_pc).T / self.resolution).astype(int)
        return xy_idx
    
    def create_layer(self, layer_name):
        self.layers[layer_name] = {'occupancy': np.zeros((self.x_size, 
                                            self.y_size), 
                                            dtype=float)}
        
    def set_layer_property(self, layer_name, property_name, value):
        '''
        Sets property to the given layer
        '''
        assert value.shape == (self.x_size, self.y_size), "layer shape does not match occupancy shape"
        self.layers[layer_name][property_name] = value
        
    def set_layer_occupancy_from_pc(self, layer_name, x_pc, y_pc, confidence):
        '''
        Sets layer to the given pointcloud in sensor frame with confidence between 0 and 1
        
        Args:
        - layer_name: name of the layer to set
        - x_pc: x positions of points in sensor frame
        - y_pc: y positions of points in sensor frame
        - confidence: confidence of each point in the pointcloud, or a single value
        '''
        # if confidence is a single value, repeat it to match the number of points
        if np.isscalar(confidence):
            confidence = np.repeat(confidence, x_pc.shape[0])
        
        assert np.min(confidence) >= 0 and np.max(confidence) <= 1, "confidence must be between 0 and 1"
        
        # convert pointcloud positions to grid indices
        xy_occ_idx = self._pc_to_grid_indices(x_pc, y_pc)
        
        # remove points that are out of bounds
        valid_indices = ((xy_occ_idx[:,0] >= 0)
                                & (xy_occ_idx[:,0] < self.x_size)
                                & (xy_occ_idx[:,1] >= 0)
                                & (xy_occ_idx[:,1] < self.y_size))
        xy_occ_idx = xy_occ_idx[valid_indices, :]
        confidence = confidence[valid_indices]
        x_occ_idx = xy_occ_idx[:,0]
        y_occ_idx = xy_occ_idx[:,1]

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

        msg.info.origin.position.x = 0.0#-self.x_origin
        msg.info.origin.position.y = -self.y_origin
        msg.data = occ_grid_vals
        
        return msg