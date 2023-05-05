'''
Overtaking Planner

Overtakes the car ahead by checking the gap
'''
import numpy as np
import copy

class Overtake:
    def __init__(self):
        self.L = 2.5
        self.time_in_overtake = 0
        self.time_in_return = 0
        self.offsets = np.array([0.7, 0.75, 0.8, 0.85, 0.9])
        self.overtake_sign = -1.0

    def check_collision(self, goal, ranges, L):
        if goal is None:
            return True
        heading_angle = np.degrees(np.arctan2(goal[1], goal[0]))
        index = int(heading_angle + 135) * 4
        votes = np.sum(np.array(ranges[index-(4*4):index+(4*4)]) < L)
        return votes > 0
    
    def do_overtake(self, waypoint_track, scan_msg):
        overtake = False

        overtake_right = [
            self.check_collision((waypoint_track[0], waypoint_track[1] + offset), scan_msg.ranges, self.L)
            for offset in -1.0 * self.offsets
        ]
        overtake_left = [
            self.check_collision((waypoint_track[0], waypoint_track[1] + offset), scan_msg.ranges, self.L)
            for offset in self.offsets
        ]
        if np.sum(overtake_right) > np.sum(overtake_left):
            self.overtake_sign = 1.0

        for offset in (self.overtake_sign * self.offsets):
            goal_shifted = copy.deepcopy(waypoint_track)
            goal_shifted[1] += offset
            if not self.check_collision(goal_shifted, scan_msg.ranges, self.L):
                overtake = True
                self.time_in_overtake += 1
                self.time_in_return = 0
                print("overtake!")
                self.L = 2.3
                goal_pos = goal_shifted
                break      
        if not overtake:
            return  
        
        return goal_shifted
        



