import numpy as np

from libf1tenth.filter import Filter


class KalmanFilter(Filter):
    
    def __init__(self, mu0, Sig0, Q, R, A, C):
        '''
        kalman filter
        
        mu0: initial state mean
        Sig0: initial state covariance
        Q: measurement noise covariance
        R: process noise covariance
        A: state transition matrix
        C: measurement matrix
        '''
        if np.isscalar(mu0):
            mu0 = np.array([mu0])
            Sig0 = np.array([Sig0]).reshape((1,1))
            Q = np.array([Q]).reshape((1,1))
            R = np.array([R]).reshape((1,1))
            A = np.array([A]).reshape((1,1))
            C = np.array([C]).reshape((1,1))
        
        self.mu = mu0
        self.Sig = Sig0
        
        self.Q = Q
        self.R = R
        self.A = A
        self.C = C
        
        self.dims = self.Sig.shape[0]
        
    def _predict(self):
        mu_kp1_k = self.A @ self.mu
        Sig_kp1_k = self.A @ self.Sig @ self.A.T + self.R
        
        return mu_kp1_k, Sig_kp1_k
    
    def _kalman_gain(self, Sig_kp1_k):
        K = Sig_kp1_k @ self.C.T @ np.linalg.inv(
            self.C @ Sig_kp1_k @ self.C.T + self.Q)
        return K
    
    def _kalman_update(self, mu_kp1_k, Sig_kp1_k, K, y_kp1):
        mu_kp1 = mu_kp1_k + K @ (y_kp1 - self.C @ mu_kp1_k)
        
        I_m_KC = (np.eye(self.dims) - K @ self.C)
        
        Sig_kp1 = I_m_KC @ Sig_kp1_k @ I_m_KC.T + K @ self.Q @ K.T
        return mu_kp1, Sig_kp1
        
    def update(self, y_kp1):
        # is y is scalar, convert to array
        if isinstance(y_kp1, (float, int)):
            y_kp1 = np.array([y_kp1])
        mu_kp1_k, Sig_kp1_k = self._predict()
        K = self._kalman_gain(Sig_kp1_k)
        self.mu, self.Sig = self._kalman_update(mu_kp1_k, Sig_kp1_k, K, y_kp1)
   
        return self
    
    def get_value(self):
        if self.dims == 1:
            return self.mu[0], self.Sig[0,0]
        else:
            return self.mu, self.Sig
    
    def is_ready(self):
        return True
        

class ExtendedKalmanFilter(Filter):
    pass


if __name__ == '__main__':
    mu = np.array([0,0,0])
    Sig0 = np.diag([1,1,1])
    Q = np.diag([1,1,1])
    R = np.diag([1,1,1])
    A = np.eye(3)
    C = np.eye(3)
    kf = KalmanFilter(
        mu0=mu, Sig0=Sig0, Q=Q, R=R, A=A, C=C)
    
    kf.update(np.array([1,1,1]))
    mu, Sig = kf.get_value()
    
    assert mu.shape == (3,)
    assert Sig.shape == (3,3)
    
    mu = 0
    Sig0 = 1
    Q = 1
    R = 1
    A = 1
    C = 1
    kf = KalmanFilter(
        mu0=mu, Sig0=Sig0, Q=Q, R=R, A=A, C=C)
    
    kf.update(1)
    mu, Sig = kf.get_value()
    
    assert np.isscalar(mu), np.isscalar(Sig)