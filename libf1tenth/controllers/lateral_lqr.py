import numpy as np
from libf1tenth.controllers import LateralController
from libf1tenth.filter import DerivativeFilter
from libf1tenth.planning.pose import Pose

class LateralLQRController(LateralController):
    def __init__(self):
        super().__init__()
        self.crosstrack_error = 0.0
        self.d_crosstrack_error = DerivativeFilter()
        self.d_crosstrack_error.update(0.0)
        pass # TODO