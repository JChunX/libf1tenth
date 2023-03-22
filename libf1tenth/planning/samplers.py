import numpy as np
from numba import njit


class Sampler:
    @staticmethod
    def sample(occupancy):
        raise NotImplementedError

class UniformSampler(Sampler):
    pass
    # @staticmethod
    # def sample(occupancy):
    #     free = np.argwhere(occupancy == 0)
    #     free_x_idx, free_y_idx = free[np.random.randint(len(free))]
    #     return free_x_idx, free_y_idx

class GaussianSampler(Sampler):
    pass
    # @staticmethod
    # def sample(occupancy, mean, std):
    #     '''
    #     Sample a point from a 2D Gaussian distribution centered at the mean
        
    #     Args:
    #     - occupancy: 2D occupancy grid
    #     - mean: 2D mean of the Gaussian distribution
    #     - std: standard deviation of the Gaussian distribution
    #     '''
    #     free = np.argwhere(occupancy == 0)
    #     free_x_idx, free_y_idx = free[np.random.randint(len(free))]
    #     return free_x_idx + np.random.normal(mean, std), free_y_idx + np.random.normal(mean, std)