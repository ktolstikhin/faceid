import uuid

from .utils.pid import PIDController


class Target:

    PID_P_GAIN = 0.8
    PID_I_GAIN = 1.5
    PID_D_GAIN = 1e-3

    def __init__(self, bbox, img_size):
        self.bbox = bbox
        self.img_size = img_size
        self.id = uuid.uuid4().hex
        self.init_controllers()

    def init_controllers(self, p=None, i=None, d=None):
        self.xmin_pid = self.controller(p, i, d)
        self.ymin_pid = self.controller(p, i, d)
        self.xmax_pid = self.controller(p, i, d)
        self.ymax_pid = self.controller(p, i, d)

    def controller(self, p=None, i=None, d=None):
        pid = PIDController(p or self.PID_P_GAIN,
                            i or self.PID_I_GAIN,
                            d or self.PID_D_GAIN)

        return pid.initialize()

    def bound_size(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w, h = self.img_size

        xmin = max(0, min(w, xmin))
        ymin = max(0, min(h, ymin))
        xmax = max(0, min(w, xmax))
        ymax = max(0, min(h, ymax))

        return [xmin, ymin, xmax, ymax]

    def correct_coordinate(self, old, new, pid):
        error = new - old

        return old + int(pid.correct(error))

    def correct_position(self, new_bbox):
        xmin_new, ymin_new, xmax_new, ymax_new = new_bbox
        xmin_old, ymin_old, xmax_old, ymax_old = self.bbox

        xmin = self.correct_coordinate(xmin_old, xmin_new, self.xmin_pid)
        ymin = self.correct_coordinate(ymin_old, ymin_new, self.ymin_pid)
        xmax = self.correct_coordinate(xmax_old, xmax_new, self.xmax_pid)
        ymax = self.correct_coordinate(ymax_old, ymax_new, self.ymax_pid)

        return [xmin, ymin, xmax, ymax]

    def update(self, bbox):
        bbox = self.correct_position(bbox)
        self.bbox = self.bound_size(bbox)

