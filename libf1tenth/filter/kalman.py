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
        self.mu = mu0
        self.Sig = Sig0
        
        self.Q = Q
        self.R = R
        self.A = A
        self.C = C
        
    def _predict(self):
        mu_kp1_k = self.A @ self.mu
        Sig_kp1_k = self.A @ self.Sig @ self.A.T + self.R
        
        return mu_kp1_k, Sig_kp1_k
    
    def _kalman_gain(self, Sig_kp1_k):
        K = Sig_kp1_k @ self.C.T @ np.linalg.inv(
            self.C @ Sig_kp1_k @ self.C.T + self.Q)
        return K
    
    def _kalman_update(self, mu_kp1_k, Sig_kp1_k, K, y_kp1):
        mu_kp1 = mu_kp1_k + K * (y_kp1 - self.C @ mu_kp1_k)
        
        I_m_KC = (np.eye(2) - K @ self.C)
        Sig_kp1 = I_m_KC @ Sig_kp1_k @ I_m_KC.T + K @ self.Q @ K.T
        return mu_kp1, Sig_kp1
        
    def update(self, y_kp1):
        mu_kp1_k, Sig_kp1_k = self._predict()
        K = self._kalman_gain(Sig_kp1_k)
        self.mu, self.Sig = self._kalman_update(mu_kp1_k, Sig_kp1_k, K, y_kp1)
   
        return self
    
    def get_value(self):
        return self.mu, self.Sig
    
    def is_ready(self):
        return True
        

class ExtendedKalmanFilter(Filter):
    pass
