import numpy as np
from libf1tenth.filter import DerivativeFilter, MovingAverageFilter
from libf1tenth.controllers import LongitudinalController

class PDSpeedController(LongitudinalController):
    def __init__(self, k_p=1.0, k_d=0.0, buffer_size=5):
        self.k_p = k_p
        self.k_d = k_d
        self.error_derivative_filter = DerivativeFilter(buffer_size=buffer_size)
        self.error_filter = MovingAverageFilter(buffer_size=buffer_size)
        
    def augment_k(self, error):
        if error < 0: # if slowing down..
            return 1.4, 0.3 # 1.2 + 0.3
        else:
            return self.k_p, self.k_d

    def control(self, speed, angle, target_speed):
        
        cur_error = target_speed - speed
        self.error_derivative_filter.update(cur_error)
        cur_error_derivative = self.error_derivative_filter.get_value()
        self.error_filter.update(cur_error)
        cur_error = self.error_filter.get_value()
        
        k_p, k_d = self.augment_k(cur_error)
        speed += k_p * cur_error + k_d * cur_error_derivative
        
        if not self.error_derivative_filter.is_ready():
            return 0.0
        speed = max(0.0, speed)
        return speed
    
if __name__ == '__main__':
    t = np.arange(0, 200)
    # create a sinusoidal profile where the frequency increses over time
    k = 20.0
    target_speeds = np.sin(t / k) * (t / k) + 1.0
    
    speed_controller = PDSpeedController(k_p=1.0, k_d=0.0, buffer_size=10)
    
    speed_history = [0.0]
    for i in range(1, len(target_speeds)):
        speed_history.append(speed_controller.control(speed=speed_history[-1], angle=0.0, target_speed=target_speeds[i]))
        
    import matplotlib.pyplot as plt
    plt.plot(speed_history)
    plt.plot(target_speeds)
    plt.show()
        