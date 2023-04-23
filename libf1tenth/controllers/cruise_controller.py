'''
Adaptive Cruise Control Controller

Follows the car ahead by adjusting target velocity to maintain a safe distance.
'''
from libf1tenth.filter import DerivativeFilter, MovingAverageFilter


class CruiseController:
    def __init__(self, k_p=0.5, k_d=0.1, 
                 cruise_control_distance=1.5,
                 buffer_size=5):
        self.k_p = k_p
        self.k_d = k_d
        self.cruise_control_distance = cruise_control_distance
        
        self.error_derivative_filter = DerivativeFilter(buffer_size=buffer_size)
        self.error_filter = MovingAverageFilter(buffer_size=buffer_size)
        
    def control(self, target_velocity, opp_car_dist):
        cur_error = opp_car_dist - self.cruise_control_distance
        self.error_derivative_filter.update(cur_error)
        cur_error_derivative = self.error_derivative_filter.get_value()
        self.error_filter.update(cur_error)
        cur_error = self.error_filter.get_value()
        
        speed = target_velocity + self.k_p * cur_error + self.k_d * cur_error_derivative
        
        if not self.error_derivative_filter.is_ready():
            return target_velocity
        
        speed = min(max(0.0, speed), target_velocity)
        
        return speed