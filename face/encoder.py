import dlib


class FaceEncoder:

    IMAGE_SIZE = 150

    def __init__(self, model_path):
        self.encoder = dlib.face_recognition_model_v1(model_path)

    def encode(self, face):
        return self.encoder.compute_face_descriptor(face)

