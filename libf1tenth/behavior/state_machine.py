import numpy as np
import time

class StateMachine:
    
    def __init__(self, states, initial_state):
        '''
        Args:
        - states: list of states
        - initial_state: initial state idx
        '''
        self.states = states
        self.current_state = initial_state
        
    def transition(self):
        pass
    
    def decode(self, idx):
        return self.states[idx]
    
    
cruise_overtaking_params = {
    'nominal_cruise_hysteresis': 0.5, # meters
    'cruise_threshold_baseline': 3, # meters, cruise distance baseline at 1m/s
    'overtake_timeout': 2.0, # seconds
    'overtake_override_velocity': 2.5, # m/s, speed at which override cruise control and overtake
}
    
    
class CruiseOvertakingStateMachine(StateMachine):
    
    def __init__(self):
        states = ['nominal', 'cruise_control', 'lane_change']
        super().__init__(states, 0)
        self.params = cruise_overtaking_params
        
        self.overtake_obs_free = False
        self.overtake_timer = 0.0
        
    def set_param(self, key, value):
        if not key in self.params:
            print('Warning: key {} not in fsm params, ignoring.'.format(key))
            return
        if not type(self.params[key]) == type(value):
            print('Warning: value {} not of type {}, ignoring.'.format(value, type(self.params[key])))
            return
        self.params[key] = value
    
    def _get_cruise_thresh(self, velocity):
        return self.params['cruise_threshold_baseline']
    
    def transition(self, telemetry):
        
        cruise_thresh = self._get_cruise_thresh(telemetry['velocity'])
        transition_flag = False
        
        if self.decode(self.current_state) == 'nominal':

            if (telemetry['wp_blocked'] 
                and telemetry['obstacle_distance'] < cruise_thresh):
                self.current_state = 1 # cruise_control
                self.cruise_timer = time.time()
                transition_flag = True
        
        elif self.decode(self.current_state) == 'cruise_control':
            if ((not telemetry['wp_blocked'])
                or (telemetry['obstacle_distance'] > cruise_thresh 
                    + self.params['nominal_cruise_hysteresis'])):
                self.current_state = 0 # nominal
                transition_flag = True
                
            elif (telemetry['in_overtake_zone'] 
                  or (telemetry['velocity'] < self.params['overtake_override_velocity'])):
                self.current_state = 2 # lane_change
                transition_flag = True
                
        elif self.decode(self.current_state) == 'lane_change':
            
            if telemetry['is_obs_free'] and not self.overtake_obs_free:
                self.overtake_timer = time.time()
                self.overtake_obs_free = True
            
            elif telemetry['is_obs_free'] and self.overtake_obs_free:
                if (time.time() - self.overtake_timer) > self.params['overtake_timeout']:
                    self.current_state = 0 # nominal
                    transition_flag = True
                    
            elif not telemetry['is_obs_free']:
                self.overtake_obs_free = False
                
        return self.current_state, transition_flag
                
                    
                