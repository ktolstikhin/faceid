from face.recognizer import FaceRecognizer
from people.detector import PeopleDetector


class PredictorFactory:

    @staticmethod
    def build(name, log=None):

        if name == 'face':
            predictor = FaceRecognizer(log)
        elif name == 'people':
            predictor = PeopleDetector(log)
        else:
            raise TypeError(f'Unknown predictor: {name}')

        return predictor

