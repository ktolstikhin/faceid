import os
import json

from .aligner import FaceAligner
from .encoder import FaceEncoder
from .detector import FaceDetector
from .classifier import FaceClassifier
from .utils.logger import init_logger
from .tracker.utils import box_center, box_in_roi
from . import settings


class FaceRecognizer:

    def __init__(self, log=None):
        self.log = log or init_logger('faceid')
        self.load_models()

    def load_models(self):
        pwd = os.getcwd()
        os.chdir(settings.model_conf_file.parent)

        with open(settings.model_conf_file) as f:
            cfg = json.load(f)

        self.aligner = FaceAligner(cfg['face_shape_predictor'], self.log)
        self.detector = FaceDetector(cfg['face_detector'],
                                     cfg['people_detector'],
                                     self.log)
        self.encoder = FaceEncoder(cfg['face_encoder'], self.log)
        self.clf = FaceClassifier(cfg['face_classifier'], self.log)

        os.chdir(pwd)

    def recognize(self, images, batch_size=32, threshold=None):
        img_faces = []
        img_face_dets = self.detector.detect_faces(images, batch_size)

        for img, face_dets in zip(images, img_face_dets):

            if not len(face_dets):
                img_faces.append([])
                continue

            face_chips = []

            for det in face_dets:
                chip = self.aligner.align(img, det)
                face_chips.append(chip)

            face_vecs = self.encoder.encode(face_chips, batch_size)
            face_ids = self.clf.predict(face_vecs, threshold, proba=True)
            body_dets = self.detector.detect_people([img])[0]

            for i, face in enumerate(face_ids):
                rect = face_dets[i].rect
                face['face_box'] = self.detector.rect_to_list(rect)

                for det in body_dets:

                    if box_in_roi(face['face_box'], det):
                        face['body_box'] = det

            img_faces.append(face_ids)

        return img_faces

