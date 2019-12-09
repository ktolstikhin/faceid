import uuid
from threading import Lock


class FaceTarget:

    def __init__(self, label, proba, face_box, body_box=None):
        self._label = label
        self._proba = proba
        self._face_box = face_box
        self._body_box = body_box
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
    def face_box(self):

        with self._lock:
            return self._face_box

    @face_box.setter
    def face_box(self, val):

        with self._lock:
            self._face_box = val

    @property
    def body_box(self):

        with self._lock:
            return self._body_box

    @body_box.setter
    def body_box(self, val):

        with self._lock:
            self._body_box = val

