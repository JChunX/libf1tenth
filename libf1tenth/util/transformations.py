import numpy as np

def to_homogenous(x, y):
    '''
    given x, y, returns homogenous coordinates of shape (3, n)
    
    x: np.ndarray of shape (n,)
    y: np.ndarray of shape (n,)
    '''
    
    return np.vstack((x.reshape(1, -1), 
                      y.reshape(1, -1), 
                      np.ones((1, x.shape[0]))))
    
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])