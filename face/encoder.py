import dlib
import numpy as np

from .utils.logger import init_logger


class FaceEncoder:

    def __init__(self, model_path, log=None):
        self.log = log or init_logger('faceid')
        self.log.info(f'Load a face encoder from {model_path}')
        self.encoder = dlib.face_recognition_model_v1(model_path)

    def encode(self, faces, batch_size=32):
        batches = round(len(faces) / batch_size) or 1
        face_vecs = []

        for i in range(batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            batch = faces[batch_start:batch_end]
            desc = self.encoder.compute_face_descriptor(batch)
            face_vecs.append(desc)

        return np.concatenate(face_vecs)

