import os

import cv2

from utils.logger import init_logger


class PopleDetector:

    DETECT_IMAGE_SIZE = (300, 300)
    DETECT_SCORE_THRES = 0.75

    def __init__(self, model_path, log=None):
        self.log = log or init_logger('faceid')
        self.log.info(f'Load a people detector from {model_path}')
        model_name = os.path.basename(model_path)
        self.detector = cv2.dnn.readNetFromTensorflow(
            os.path.join(model_path, model_name + '.pb'),
            os.path.join(model_path, model_name + '.pbtxt')
        )

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
        w_scale, h_scale = images[0].shape[1] / w, images[0].shape[0] / h
        person_index = 1
        people_dets = []

        for dets in image_dets:
            person_boxes = []

            for det in dets:
                class_index, class_score = det[1], det[2]

                if class_index != person_index:
                    continue

                if class_score < self.DETECT_SCORE_THRES:
                    continue

                xmin = int(det[3] * w * w_scale)
                ymin = int(det[4] * h * h_scale)
                xmax = int(det[5] * w * w_scale)
                ymax = int(det[6] * h * h_scale)
                person_boxes.append([xmin, ymin, xmax, ymax])

            people_dets.append(person_boxes)

        return people_dets
