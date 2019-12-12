import ctypes
from multiprocessing import Process, Array, Event

import numpy as np

from .stream import VideoStream
from utils.logger import init_logger


class VideoStreamProcess(Process):

    def __init__(self, path='/dev/video0', size=(640, 480), log=None):
        self._stream = VideoStream(path, size)
        self._join_event = Event()
        self._log = log or init_logger('faceid')
        self._frame_shape = (size[1], size[0], 3)
        array_size = int(np.prod(self._frame_shape))
        self._frame = Array(ctypes.c_int8, array_size)
        super().__init__(name='VideoStream')

    @property
    def size(self):
        return self._stream.size

    @property
    def path(self):
        return self._stream.path

    def _shared_to_numpy(self):
        arr = np.frombuffer(self._frame.get_obj(), dtype='uint8')

        return arr.reshape(self._frame_shape)

    def read(self):

        with self._frame.get_lock():
            return self._shared_to_numpy()

    def run(self):
        self._log.info(f'Start fetching frames from {self.path}')

        while not self._join_event.is_set():
            frame = self._stream.read()

            if frame is None:
                continue

            with self._frame.get_lock():
                arr = self._shared_to_numpy()
                np.copyto(arr, frame)

    def join(self, timeout=None):
        self._log.info(f'Stopping {self.name} process...')
        self._join_event.set()
        super().join(timeout)

