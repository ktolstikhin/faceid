import os

import cv2
import dlib
import numpy as np

from .utils.logger import init_logger


class FaceDetector:

    DETECT_IMAGE_SIZE = (300, 300)
    DETECT_SCORE_THRES = 0.75

    def __init__(self, face_model_path, people_model_path, log=None):
        self.log = log or init_logger('faceid')
        self.log.info(f'Load a face detector from {face_model_path}')
        self.face_detector = dlib.cnn_face_detection_model_v1(face_model_path)
        self.log.info(f'Load a people detector from {people_model_path}')
        model_name = os.path.basename(people_model_path)
        self.people_detector = cv2.dnn.readNetFromTensorflow(
            os.path.join(people_model_path, model_name + '.pb'),
            os.path.join(people_model_path, model_name + '.pbtxt')
        )

    def detect_faces(self, images, batch_size=32, upsample=1):
        batches = round(len(images) / batch_size) or 1
        face_dets = []

        for i in range(batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            batch = images[batch_start:batch_end]
            det = self.face_detector(batch, upsample)
            face_dets.append(det)

        return np.concatenate(face_dets)

    def detect_people(self, images):
        image_blob = cv2.dnn.blobFromImages(images,
                                            size=self.DETECT_IMAGE_SIZE,
                                            mean=(127.5, 127.5, 127.5))
        self.people_detector.setInput(image_blob)
        image_dets = self.people_detector.forward()

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

    def rect_to_list(self, rect):
        return [rect.left(), rect.top(), rect.right(), rect.bottom()]

    def list_to_rect(self, box):
        xmin, ymin, xmax, ymax = box

        return dlib.rectangle(left=xmin, top=ymin, right=xmax, bottom=ymax)

