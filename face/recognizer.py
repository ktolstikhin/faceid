import os
import json

from .aligner import FaceAligner
from .encoder import FaceEncoder
from .detector import FaceDetector
from .classifier import FaceClassifier
from .utils.logger import init_logger
from .tracker.utils import box_center, point_in_roi
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
        self.detector = FaceDetector(cfg['face_detector'], self.log)
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

            for i, face in enumerate(face_ids):
                face['face_box'] = self.detector.to_list(face_dets[i].rect)

            img_faces.append(face_ids)

        img_people_dets = self.detector.detect_people(images)

        for faces, body_boxes in zip(img_faces, img_people_dets):

            for face in faces:
                face_center = box_center(face['face_box'])

                for body_box in body_boxes:

                    if point_in_roi(face_center, body_box):
                        face['body_box'] = body_box

        return img_faces

