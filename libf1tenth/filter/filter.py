import numpy as np


class Filter:
    '''
    filter base class
    
    usage:
    
    filter = Filter() # use a specific filter
    filter.update(value)
    value = filter.get_value()
    
    '''
    
    def __init__(self):
        pass
    
    def update(self, value):
        return self
    
    def get_value(self):
        raise NotImplementedError
    
    def is_ready(self):
        return False