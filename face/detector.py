import dlib
import numpy as np

from utils.logger import init_logger


class FaceDetector:

    def __init__(self, model_path, log=None):
        self.log = log or init_logger('faceid')
        self.log.info(f'Load a face detector from {model_path}')
        self.detector = dlib.cnn_face_detection_model_v1(model_path)

    def detect(self, images, batch_size=32, upsample=1):
        batches = round(len(images) / batch_size) or 1
        face_dets = []

        for i in range(batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            batch = images[batch_start:batch_end]
            det = self.detector(batch, upsample)
            face_dets.append(det)

        return np.concatenate(face_dets)

    def to_list(self, rect):
        return [rect.left(), rect.top(), rect.right(), rect.bottom()]

    def to_rect(self, box):
        xmin, ymin, xmax, ymax = box

        return dlib.rectangle(left=xmin, top=ymin, right=xmax, bottom=ymax)

