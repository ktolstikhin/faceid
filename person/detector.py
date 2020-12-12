import os
import json
import logging

import cv2
import numpy as np

from cfg import settings
from vision.predictor.abc import Predictor


class PersonDetector(Predictor):

    DETECT_IMAGE_SIZE = (300, 300)
    DETECT_PROBA_THRES = 0.75

    def __init__(self):
        self.load_models()

    @property
    def log(self):
        return logging.getLogger(settings.logger)

    def load_models(self):
        pwd = os.getcwd()
        os.chdir(settings.model_conf_file.parent)

        with open(settings.model_conf_file) as f:
            cfg = json.load(f)

        model_path = cfg['person_detector']
        self.log.info(f'Load a person detector from {model_path}')

        model_name = os.path.basename(model_path)
        self.detector = cv2.dnn.readNetFromTensorflow(
            os.path.join(model_path, f'{model_name}.pb'),
            os.path.join(model_path, f'{model_name}.pbtxt')
        )

        os.chdir(pwd)

    def detect(self, images):
        image_blob = cv2.dnn.blobFromImages(images,
                                            size=self.DETECT_IMAGE_SIZE,
                                            mean=(127.5, 127.5, 127.5))
        self.detector.setInput(image_blob)
        image_dets = self.detector.forward()

        try:
            image_dets = image_dets.reshape((len(images), 100, 7))
        except ValueError:
            return [[] for _ in range(len(images))]

        w, h = self.DETECT_IMAGE_SIZE
        img_h, img_w = images[0].shape[:2]
        w_scale, h_scale = img_w / w, img_h / h

        person_index = 1
        image_persons = []

        for dets in image_dets:
            persons = []

            for det in dets:
                class_index, class_proba = det[1], det[2]

                if class_index != person_index:
                    continue

                if class_proba < self.DETECT_PROBA_THRES:
                    continue

                xmin = int(det[3] * w * w_scale)
                ymin = int(det[4] * h * h_scale)
                xmax = int(det[5] * w * w_scale)
                ymax = int(det[6] * h * h_scale)

                persons.append({
                    'label': 'person',
                    'proba': class_proba,
                    'box': [xmin, ymin, xmax, ymax]
                })

            image_persons.append(persons)

        return image_persons

    def predict(self, images, batch_size=32):
        batches = round(len(images) / batch_size) or 1
        predictions = []

        for i in range(batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            batch = images[batch_start:batch_end]
            batch_preds = self.detect(batch)
            predictions.append(batch_preds)

        return [i for batch in predictions for i in batch]

