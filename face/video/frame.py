from threading import Lock

from .stream import VideoStream


class FrameBuffer:

    _frames = {}
    _lock = Lock()

    def add(self, frame, targets=None, title=None):

        if targets is not None:

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

        if VideoStream.is_key_pressed('q'):
            raise KeyboardInterrupt

