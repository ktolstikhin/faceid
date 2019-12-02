import dlib

from .utils.logger import init_logger


class FaceEncoder:

    IMAGE_SIZE = 150

    def __init__(self, model_path, log=None):
        self.log = log or init_logger('faceid')
        self.log.info(f'Load a face encoder from {model_path}')
        self.encoder = dlib.face_recognition_model_v1(model_path)

    def encode(self, face):
        return self.encoder.compute_face_descriptor(face)

