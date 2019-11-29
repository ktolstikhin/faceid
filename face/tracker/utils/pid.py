import time
import math


class PIDController(object):

    def __init__(self, p_gain, i_gain, d_gain, i_max=None, d_max=None):
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.i_max = i_max
        self.d_max = d_max

    def _bound_value(self, val, max_val):

        if max_val is not None and abs(val) > max_val:
            return math.copysign(max_val, val)
        else:
            return val

    def initialize(self):
        self.prev_time = time.time()
        self.prev_error = 0

        self.p_term = 0
        self.i_term = 0
        self.d_term = 0

        return self

    def correct(self, error):
        self.p_term = error

        cur_time = time.time()
        time_delta = cur_time - self.prev_time

        i_term = self.i_term + error * time_delta
        self.i_term = self._bound_value(i_term, self.i_max)

        if time_delta > 0:
            d_term = (error - self.prev_error) / time_delta
            self.d_term = self._bound_value(d_term, self.d_max)
        else:
            self.d_term = 0

        output = (
            self.p_gain * self.p_term
            + self.i_gain * self.i_term
            + self.d_gain * self.d_term
        )

        self.prev_time = cur_time
        self.prev_error = error

        return output

