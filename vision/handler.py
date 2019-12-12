import queue
from threading import Thread, Event

from .predictor.factory import PredictorFactory
from utils.logger import init_logger


class VisionTaskHandler(Thread):

    QUEUE_GET_TIMEOUT = 5

    def __init__(self, name, task_queue, batch_size=32, log=None):
        self.log = log or init_logger('faceid')
        self.predictor = PredictorFactory.build(name, self.log)
        self.task_queue = task_queue
        self.batch_size = batch_size
        self.join_event = Event()
        super().__init__(name='VisionTaskHandler')

    def run(self):
        self.log.info('Start handling vision tasks...')

        while not self.join_event.is_set():
            tasks = []
            block = True

            for _ in range(self.batch_size):

                try:
                    task = self.task_queue.get(block, self.QUEUE_GET_TIMEOUT)
                    tasks.append(task)
                    block = False
                except queue.Empty:
                    break

            if not tasks:
                continue

            images = [t.image for t in tasks]
            predictions = self.predictor.predict(images, self.batch_size)

            for i, task in enumerate(tasks):
                task.results = predictions[i]
                self.task_queue.task_done()

    def join(self, timeout=None):
        self.log.info(f'Stopping {self.name} thread...')
        self.task_queue.join()
        self.join_event.set()
        super().join(timeout)

