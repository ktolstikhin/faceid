from copy import copy
from threading import Lock
from collections import defaultdict


class FaceTargetKeeper:

    _targets = {}
    _faces = defaultdict(dict)
    _lock = Lock()

    def add(self, target):

        with self._lock:
            self._targets[target.id] = target
            self._faces[target.label][target.id] = target

    def remove(self, target):

        with self._lock:

            try:
                del self._targets[target.id]
                del self._faces[target.label][target.id]
            except KeyError:
                pass

    def get(self, label=None, target_id=None):

        with self._lock:

            if label is None and target_id is None:
                return {l: copy(d) for l, d in self._faces.items()}

            if target_id is None:
                return copy(self._faces.get(label))
            else:
                return self._targets.get(target_id)

