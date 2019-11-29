import dlib


class FaceAligner:

    def __init__(self, model_path):
        self.shape_predictor = dlib.shape_predictor(model_path)

    def align(self, img, face, size=150, padding=0.25):
        shape = self.shape_predictor(img, face.rect)
        face_chip = dlib.get_face_chip(img, shape, size, padding)

        return face_chip

