import os
import json

from .aligner import FaceAligner
from .encoder import FaceEncoder
from .detector import FaceDetector
from .classifier import FaceClassifier
from vision.predictor.abc import Predictor
from utils.logger import init_logger
from cfg import settings


class FaceRecognizer(Predictor):

    def __init__(self, log=None):
        self.log = log or init_logger('faceid')
        self.load_models()

    def load_models(self):
        pwd = os.getcwd()
        os.chdir(settings.model_conf_file.parent)

        with open(settings.model_conf_file) as f:
            cfg = json.load(f)

        self.aligner = FaceAligner(cfg['face_shape_predictor'], self.log)
        self.detector = FaceDetector(cfg['face_detector'], self.log)
        self.encoder = FaceEncoder(cfg['face_encoder'], self.log)
        self.clf = FaceClassifier(cfg['face_classifier'], self.log)

        os.chdir(pwd)

    def predict(self, images, batch_size=32, threshold=None):
        faces = []
        face_dets = self.detector.detect(images, batch_size)

        for img, dets in zip(images, face_dets):

            if not len(dets):
                faces.append([])
                continue

            face_chips = []

            for det in dets:
                chip = self.aligner.align(img, det)
                face_chips.append(chip)

            face_vecs = self.encoder.encode(face_chips, batch_size)
            face_ids = self.clf.predict(face_vecs, threshold, proba=True)

            for i, face in enumerate(face_ids):
                face['box'] = self.detector.to_list(dets[i].rect)

            faces.append(face_ids)

        return faces

