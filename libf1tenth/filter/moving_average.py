import numpy as np

from libf1tenth.filter import Filter


class MovingAverageFilter(Filter):
    def __init__(self, buffer_size=3):
        assert buffer_size >= 1, 'buffer_size must be >= 1'
        assert isinstance(buffer_size, int), 'buffer_size must be an integer'
        self.buffer_size = buffer_size
        self.buffer = []

    def update(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        return self
    
    def get_value(self):
        return np.mean(self.buffer)
    
    def is_ready(self):
        return len(self.buffer) == self.buffer_size