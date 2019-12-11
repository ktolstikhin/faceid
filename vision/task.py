from threading import Condition


class VisionTask:

    def __init__(self, image=None):
        self.image = image
        self.done = Condition()
        self._results = None

    @property
    def results(self):

        with self.done:
            self.done.wait()

            return self._results

    @results.setter
    def results(self, predictions):

        with self.done:
            self._results = predictions
            self.done.notify()

