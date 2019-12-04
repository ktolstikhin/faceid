from threading import Condition


class VisionTask:

    def __init__(self, image=None):
        self.image = image
        self.done = Condition()
        self._faces = None

    @property
    def faces(self):

        with self.done:
            self.done.wait()

            return self._faces

    @faces.setter
    def faces(self, predictions):

        with self.done:
            self._faces = predictions
            self.done.notify()

