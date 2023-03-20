import numpy as np
from numba import njit


@njit
def query_euclidean_distance(values, query):
    '''
    Returns euclidean distance between a query and a set of values
    
    Args:
    - values: ndarray of shape (d, n)
    - query: ndarray of shape (d,)
    '''
    distance = np.linalg.norm(values - query)
    return distance