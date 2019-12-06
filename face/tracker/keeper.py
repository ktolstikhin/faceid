from threading import Lock
from collections import defaultdict


class FaceTargetKeeper:

    _targets = {}
    _faces = defaultdict(dict)
    _lock = Lock()

    @staticmethod
    def add(target):

        with self._lock:
            self._targets[target.id] = target
            self._faces[target.label][target.id] = target

    @staticmethod
    def remove(target):

        with self._lock:

            try:
                del self._targets[target.id]
                del self._faces[target.label][target.id]
            except KeyError:
                pass

    @staticmethod
    def get(label=None, target_id=None):

        with self._lock:

            if label is None and target_id is None:
                return self._faces

            if target_id is None:
                return self._faces.get(label)
            else:
                return self._targets.get(target_id)

