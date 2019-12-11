from face.recognizer import FaceRecognizer
from person.detector import PersonDetector


class PredictorFactory:

    @staticmethod
    def build(name, log=None):

        if name == 'face':
            predictor = FaceRecognizer(log)
        elif name == 'person':
            predictor = PersonDetector(log)
        else:
            raise ValueError(f'Unknown predictor: {name}')

        return predictor

