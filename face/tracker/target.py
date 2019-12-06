import uuid
from threading import Lock


class FaceTarget:

    def __init__(self, label, proba, box):
        self.label = label
        self.proba = proba
        self._box = box
        self._lock = Lock()
        self.id = uuid.uuid4().hex

    @property
    def box(self):

        with self._lock:
            return self._box

    @box.setter
    def box(self, val):

        with self._lock:
            self._box = val

