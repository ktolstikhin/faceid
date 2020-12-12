from face.recognizer import FaceRecognizer
from person.detector import PersonDetector


class PredictorFactory:

    @staticmethod
    def build(name):

        if name == 'face':
            predictor = FaceRecognizer()
        elif name == 'person':
            predictor = PersonDetector()
        else:
            raise ValueError(f'Unknown predictor: {name}')

        return predictor

