import os
import uuid
from threading import Lock

from .stream import VideoStream
from utils.logger import init_logger


class FrameBuffer:

    _frames = {}
    _lock = Lock()

    def __init__(self, img_dir=None, log=None):
        self.img_dir = img_dir or '.'
        self.log = log or init_logger('faceid')

    def add(self, frame, targets=None, title=None):

        if targets is None:
            targets = []

        for t in targets:
            text = f'{t.label}: {t.proba:.2f}'
            anchor = (t.box[0], t.box[1] - 5)
            VideoStream.draw_text(frame, text, anchor)
            VideoStream.draw_box(frame, t.box)

        with self._lock:
            self._frames[title] = frame

    def show(self):

        with self._lock:
            frames = list(self._frames.items())

        for title, frame in frames:
            VideoStream.show(frame, title)

        char = VideoStream.wait_key()

        if char == 'q':
            VideoStream.close_windows()
            raise KeyboardInterrupt

        if char == 's':
            filename = f'{uuid.uuid4().hex}.jpg'
            filepath = os.path.join(self.img_dir, filename)
            VideoStream.save(frame, filepath)
            self.log.info(f'Save snapshot to {filepath}')

