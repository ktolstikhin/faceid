import uuid
from threading import Lock


class Target:

    def __init__(self, label, proba, box):
        self._label = label
        self._proba = proba
        self._box = box
        self._lock = Lock()
        self._id = uuid.uuid4().hex

    @property
    def id(self):
        return self._id

    @property
    def label(self):

        with self._lock:
            return self._label

    @label.setter
    def label(self, val):

        with self._lock:
            self._label = val

    @property
    def proba(self):

        with self._lock:
            return self._proba

    @proba.setter
    def proba(self, val):

        with self._lock:
            self._proba = val

    @property
    def box(self):

        with self._lock:
            return self._box

    @box.setter
    def box(self, val):

        with self._lock:
            self._box = val

