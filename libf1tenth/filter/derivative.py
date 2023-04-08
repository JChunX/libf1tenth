import numpy as np

from libf1tenth.filter.moving_average import MovingAverageFilter
from libf1tenth.filter import Filter

class DerivativeFilter(Filter):
    '''
    computes finite difference derivative of a moving average
    '''
    
    def __init__(self, buffer_size=3):
        self.average_filter = MovingAverageFilter(buffer_size)
        
    def update(self, value):
        self.average_filter.update(value)
        return self
    
    def get_value(self):
        buffer = self.average_filter.buffer
        if len(buffer) < 2:
            return 0.0
        return np.mean(np.diff(buffer))
    
    def is_ready(self):
        return self.average_filter.is_ready()
    
if __name__ == '__main__':
    # create sinusoidal profile
    t = np.arange(0, 200)
    y = np.sin(t / 20.0) * (t / 20.0) + 1.0
    filter = DerivativeFilter(buffer_size=10)
    dy = []
    for i in range(len(y)):
        y[i] += np.random.normal(0, 0.1)
        filter.update(y[i])
        dy.append(filter.get_value())
        
    import matplotlib.pyplot as plt
    plt.plot(y)
    plt.plot(dy)
    plt.show()
        