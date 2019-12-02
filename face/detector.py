import dlib

from .utils.logger import init_logger


class FaceDetector:

    def __init__(self, model_path, log=None):
        self.log = log or init_logger('faceid')
        self.log.info(f'Load a face detector from {model_path}')
        self.detector = dlib.cnn_face_detection_model_v1(model_path)

    def detect(self, img, upsample=1):
        return self.detector(img, upsample)

    def to_list(self, rect):
        return [rect.left(), rect.top(), rect.right(), rect.bottom()]

    def to_rect(self, box):
        xmin, ymin, xmax, ymax = box

        return dlib.rectangle(left=xmin, top=ymin, right=xmax, bottom=ymax)

