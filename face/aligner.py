import dlib

from utils.logger import init_logger


class FaceAligner:

    def __init__(self, model_path, log=None):
        self.log = log or init_logger('faceid')
        self.log.info(f'Load a face shape predictor from {model_path}')
        self.shape_predictor = dlib.shape_predictor(model_path)

    def align(self, img, face, size=150, padding=0.25):
        shape = self.shape_predictor(img, face.rect)
        face_chip = dlib.get_face_chip(img, shape, size, padding)

        return face_chip

