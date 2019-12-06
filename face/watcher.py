from threading import Thread, Event

from .vision import VisionTask
from .tracker import FaceTracker
from .video.frame import FrameBuffer
from .video.utils import create_stream
from .utils.logger import init_logger


class FaceWatcher(Thread):

    def __init__(self, task_queue, video_conf, show=False, log=None):
        self.task_queue = task_queue
        self.stream = create_stream(video_conf)
        self.log = log or init_logger('faceid')
        self.log.info(f'Start video stream from {self.stream.path}')
        self.frame_buffer = FrameBuffer() if show else None
        self.tracker = FaceTracker(self.stream.size)
        self.join_event = Event()
        super().__init__(name='FaceWatcher')

    def run(self):
        self.log.info('Start watching faces...')
        task = VisionTask()

        while not self.join_event.is_set():
            frame = self.stream.read()

            if frame is None:
                self.log.warning(f'Failed to read from {self.stream.path}')
                continue

            task.image = frame
            self.task_queue.put(task)
            self.tracker.update(task.faces)
            targets = self.tracker.get_targets()

            if self.frame_buffer is not None:
                self.frame_buffer.add(frame, targets, self.stream.path)

    def join(self, timeout=None):
        self.log.info(f'Stopping {self.name} thread...')
        self.join_event.set()
        super().join(timeout)

