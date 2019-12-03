import uuid
from threading import Lock


class Target:

    def __init__(self, bbox):
        self._bbox = bbox
        self.lock = Lock()
        self.id = uuid.uuid4().hex

    @property
    def bbox(self):

        with self.lock:
            return self._bbox

    @bbox.setter
    def bbox(self, val):

        with self.lock:
            self._bbox = val

