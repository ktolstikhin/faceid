import json
from threading import Thread, Event

from .vision import VisionTask
from .utils.logger import init_logger
from .video import VideoDeviceSettings, VideoStream


class FaceWatcher(Thread):

    def __init__(self, task_queue, video_conf, show=False, log=None):
        self.task_queue = task_queue
        self.log = log or init_logger('faceid')
        self.stream = self.video_stream(video_conf)
        self.show = show
        self.join_event = Event()
        super().__init__(name='FaceWatcher')

    def video_stream(self, conf_file):

        with open(conf_file) as f:
            cfg = json.load(f)

        path, size = cfg['path'], tuple(cfg['resolution'])
        self.log.info(f'Initialize video stream {path}')
        stream = VideoStream(path, size)
        s = cfg.get('settings')

        if s is not None:
            self.log.info(f'Apply video stream settings...')
            vds = VideoDeviceSettings(path)
            vds.exposure_manual()
            vds.set(s)

        return stream

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
            faces = task.faces
            self.log.info(f'Detected {len(faces)} faces')

            if self.show:
                # TODO: track face targets
                targets = []
                self.show_targets(frame, targets)

    def show_targets(self, frame, targets):

        for t in targets:
            anchor = (t.bbox[0], t.bbox[1] - 5)
            VideoStream.draw_text(frame, t.label, anchor)
            VideoStream.draw_box(frame, t.bbox)

        VideoStream.show(frame, title=self.stream.path)

        if VideoStream.is_key_pressed('q'):
            self.join_event.set()

    def join(self, timeout=None):
        self.log.info(f'Stopping {self.name} thread...')
        self.join_event.set()
        super().join(timeout)

