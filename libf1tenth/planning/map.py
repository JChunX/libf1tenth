import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml


class Map:
    '''
    wraps ros2 map into a class
    '''
    def __init__(self):
        self.yaml_path = ''
        self.image_path = ''
        
        self.image = None
        self.resolution = None
        self.origin = [0.0, 0.0, 0.0]
        self.negate = 0
        self.occupied_thresh = 0.65
        self.free_thresh = 0.196
    
    
    def initialize(self, path):
        '''
        initializes the map object from a yaml file
        '''
        # read yaml file into a dictionary
        params_dict = {}
        params_dict = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
        
        self.yaml_path = path
        self.image_path = os.path.join(os.path.dirname(path), 
                                       params_dict['image'])
        self.resolution = params_dict['resolution']
        self.origin = params_dict['origin']
        self.negate = params_dict['negate']
        self.occupied_thresh = params_dict['occupied_thresh']
        self.free_thresh = params_dict['free_thresh']
        
        # read image
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        
    def show_map(self):
        '''
        shows the map
        '''
        plt.figure()
        plt.imshow(self.image, cmap='gray')
        
    def show_coordinates(self, xs_cart, ys_cart, color=None, size=600):
        '''
        shows the map with a coordinate marked in cartesian coordinates
        xs: ndarray of x coordinates
        ys: ndarray of y coordinates
        color: ndarray of colors
        size: size of the map in pixels
        '''
        # if x, y are not iterables, convert them to lists
        if not hasattr(xs_cart, '__iter__'):
            xs_cart = np.array([xs_cart])
            ys_cart = np.array([ys_cart])
        
        half_size = int(size / 2)
        plt.figure(figsize=(30, 30))
        # use initial coordinates to center the map
        
        xs, ys = self.cartesian_to_pixel(xs_cart, -ys_cart) # -ys flips the y axis for correct plotting
        x0 = xs[0]
        y0 = ys[0]
        
        plt.imshow(self.image[y0-half_size:y0+half_size, x0-half_size:x0+half_size], cmap='gray')
        
        if color is not None:
            plt.scatter(xs-x0+half_size, ys-y0+half_size, s=100, c=color, cmap='rainbow')
        else:
            plt.scatter(xs-x0+half_size, ys-y0+half_size, s=100, c=np.arange(len(xs)), cmap='rainbow')
            
        step = 1 # meters
        step = step / self.resolution # convert to pixels
        
        xtick_origin = half_size + xs_cart[0] / self.resolution
        ytick_origin = half_size + ys_cart[0] / self.resolution
        xtick_neg = np.arange(xtick_origin, xtick_origin-half_size, -step)
        xtick_pos = np.arange(xtick_origin, xtick_origin+half_size, step)
        ytick_neg = np.arange(ytick_origin, ytick_origin-half_size, -step)
        ytick_pos = np.arange(ytick_origin, ytick_origin+half_size, step)
        
        xticks = np.concatenate((xtick_neg[::-1][:-1], xtick_pos))
        yticks = np.concatenate((ytick_neg[::-1][:-1], ytick_pos))
        
        xlabels_pos = np.arange(0, len(xtick_pos))
        xlabels_neg = np.arange(0, -len(xtick_neg), -1)
        ylabels_pos = np.arange(0, -len(ytick_pos),-1)
        ylabels_neg = np.arange(0, len(ytick_neg))
        
        xlabels = np.concatenate((xlabels_neg[::-1][:-1], xlabels_pos))
        ylabels = np.concatenate((ylabels_neg[::-1][:-1], ylabels_pos))
        
        plt.xticks(xticks, xlabels, rotation=45, fontsize=20)
        plt.yticks(yticks, ylabels, rotation=0, fontsize=20)
        
        plt.grid()
        plt.show()
        
    def cartesian_to_pixel(self, x, y):
        '''
        converts cartesian coordinates to pixel coordinates
        '''
        x_pixel = np.round((x - self.origin[0]) / self.resolution).astype(int)
        y_pixel = np.round((y - self.origin[1]) / self.resolution).astype(int)
        
        return x_pixel, y_pixel