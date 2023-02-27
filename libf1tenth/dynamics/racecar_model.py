import json
from pathlib import Path

from libf1tenth.dynamics.car_dynamics import vehicle_dynamics_st


class RaceCarModel:
    
    def __init__(self):
        # relative import https://stackoverflow.com/questions/40416072/reading-a-file-using-a-relative-path-in-a-python-project
        dynamics_params_path = Path(__file__).parent / "../config/dynamics.json"    
        with open(dynamics_params_path, 'r') as f:
            params = json.load(f)
        self.mu = params['mu'] # friction coefficient
        self.C_Sf = params['C_Sf'] # cornering stiffness front
        self.C_Sr = params['C_Sr'] # cornering stiffness rear
        self.lf = params['lf'] # distance from CG to front axle
        self.lr = params['lr'] # distance from CG to rear axle
        self.h = params['h'] # distance from CG to ground
        self.m = params['m'] # vehicle mass
        self.I = params['I'] # vehicle moment of inertia

        #steering constraints
        self.s_min = params['s_min']  #minimum steering angle [rad]
        self.s_max = params['s_max']  #maximum steering angle [rad]
        self.sv_min = params['sv_min']  #minimum steering velocity [rad/s]
        self.sv_max = params['sv_max']  #maximum steering velocity [rad/s]

        #longitudinal constraints
        self.v_switch = params['v_switch']  #switching velocity [m/s]
        self.a_max = params['a_max']  #maximum acceleration [m/s^2]
        self.v_min = params['v_min']  #minimum velocity [m/s]
        self.v_max = params['v_max']  #maximum velocity [m/s]
    
    def evaluate_dynamics(self, x, u):
        # single track dynamic model
        return vehicle_dynamics_st(x, u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)
    