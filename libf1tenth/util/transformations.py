import numpy as np


def to_homogenous(x, y):
    '''
    given x, y, returns homogenous coordinates of shape (3, n)
    
    x: np.ndarray of shape (n,)
    y: np.ndarray of shape (n,)
    '''
    if np.isscalar(x) and np.isscalar(y):
        return np.array([x,y,1]).reshape(3,1)
    
    return np.vstack((x.reshape(1, -1), 
                      y.reshape(1, -1), 
                      np.ones((1, x.shape[0]))))
    
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def coordinate_transform_ba(x_a, theta_ba, p_ba):
    '''
    Given a point in frame a, and transformations from b to a, 
    returns the point in frame b
    
    Args:
    - x_a: np.ndarray of shape (2, n), representing a point in frame 'a'
    - theta_ba: float, representing a rotation that takes frame 'b' to frame 'a'
    - p_ba: np.ndarray of shape (2, 1), representing a translation that takes frame 'b' to frame 'a'
    '''

    R_ba = rotation_matrix(theta_ba)
    x_b = R_ba @ x_a + p_ba
    return x_b
    
def coordinate_transform_ab(x_b, theta_ba, p_ba):
    '''
    Given a point in frame b, and transformations from b to a,
    returns the point in frame a
    
    Args:
    - x_b: np.ndarray of shape (2, n), representing a point in frame 'b'
    - theta_ba: float, representing a rotation that takes frame 'b' to frame 'a'
    - p_ba: np.ndarray of shape (2, 1), representing a translation that takes frame 'b' to frame 'a'
    '''
    
    R_ba = rotation_matrix(theta_ba)
    x_a = R_ba.T @ (x_b - p_ba)
    return x_a