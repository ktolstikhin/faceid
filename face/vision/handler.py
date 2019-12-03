import queue
from threading import Thread

from ..utils.logger import init_logger


class VisionTaskHandler(Thread):

    def __init__(self, task_queue, batch_size=32, log=None):
        self.task_queue = task_queue
        self.batch_size = batch_size
        self.log = log or init_logger('faceid')
        self.recognizer = FaceRecognizer(self.log)
        super().__init__()

    def run(self):
        self.log.info('Start handling vision tasks...')
        stop = False

        while not stop:
            tasks = []
            block = True

            for _ in range(self.batch_size):

                try:
                    task = self.task_queue.get(block)
                    tasks.append(task)
                    block = False
                except queue.Empty:
                    break

            if None in tasks:
                stop = True
                self.log.info('Received a stop signal. Shutting down...')

                if len(tasks) == 1:
                    break

                tasks = [t for t in tasks if t is not None]

            images = [t['image'] for t in tasks]
            faces = self.recognizer.recognize(images, self.batch_size)

            for i, task in enumerate(tasks):
                task.faces = faces[i]

