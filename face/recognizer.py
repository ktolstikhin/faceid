import os
import json

from .aligner import FaceAligner
from .encoder import FaceEncoder
from .detector import FaceDetector
from .utils.logger import init_logger


class FaceRecognizer:

    CONF_DIR = 'cfg'
    MODEL_CONF_FILE = 'models.json'

    def __init__(self, log=None):
        self.log = log or init_logger('faceid')
        self.load_models()

    def load_models(self):
        cfg_dir = os.path.join(os.path.dirname(__file__), self.CONF_DIR)
        pwd = os.getcwd()
        os.chdir(cfg_dir)

        with open(self.MODEL_CONF_FILE) as f:
            cfg = json.load(f)

        self.aligner = FaceAligner(cfg['face_shape_predictor'])
        self.detector = FaceDetector(cfg['face_detector'])
        self.encoder = FaceEncoder(cfg['face_encoder'])
        self.clf = FaceClassifier(cfg['face_classifier'], self.log)

        os.chdir(pwd)

    def recognize(self, images, thres=None):
        self.log.info(f'Received {len(images)} images.')
        faces = []

        for i, img in enumerate(images, start=1):
            face_dets = self.detector.detect(img)
            self.log.info(f'Image #{i}: Detected {len(face_dets)} faces.')

            if not face_dets:
                faces.append([])
                continue

            face_chips = []

            for det in face_dets:
                chip = self.aligner.align(img, det)
                face_chips.append(chip)

            face_vecs = self.encoder.encode(face_chips)
            face_ids = self.clf.predict(face_vecs, proba=True, threshold=thres)

            unknown = self.clf.UNKNOWN_FACE_LABEL
            rec_num = sum(1 for face in face_ids if face['label'] != unknown)
            self.log.info(f'Image #{i}: Recognized {rec_num} faces.')

            for j, face in enumerate(face_ids):
                face['box'] = self.detector.to_list(face_dets[j].rect)

            faces.append(face_ids)

        return faces

