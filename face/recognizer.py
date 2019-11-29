import os
import json

from .aligner import FaceAligner
from .encoder import FaceEncoder
from .detector import FaceDetector
from .utils.logger import init_logger


class FaceRecognizer:

    CONF_FILE = 'models.json'

    def __init__(self, log=None):
        self.log = log or init_logger('faceid')
        self.load_models()

    def load_models(self):
        cfg_dir = os.path.join(os.path.dirname(__file__), 'cfg')
        pwd = os.getcwd()
        os.chdir(cfg_dir)

        with open(self.CONF_FILE) as f:
            cfg = json.load(f)

        self.aligner = FaceAligner(cfg['face_shape_predictor'])
        self.detector = FaceDetector(cfg['face_detector'])
        self.encoder = FaceEncoder(cfg['face_encoder'])

        os.chdir(pwd)

