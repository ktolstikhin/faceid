import cv2
import dlib
import numpy as np

from .utils.logger import init_logger
from .utils.nonmaxsup import merge_boxes


class FaceDetector:

    def __init__(self, face_detector, log=None):
        self.log = log or init_logger('faceid')
        self.log.info(f'Load a face detector from {face_detector}')
        self.face_detector = dlib.cnn_face_detection_model_v1(face_detector)
        self.people_detector = cv2.HOGDescriptor()
        self.people_detector.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector())

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
        people_dets = []

        for img in images:
            rects = self.people_detector.detectMultiScale(
                img, winStride=(4, 4), padding=(8, 8), scale=1.05)[0]
            det = [[x, y, x + w, y + h] for x, y, w, h in rects]
            det = merge_boxes(det, thres=0.3)
            people_dets.append(det)

        return people_dets

    def to_list(self, rect):
        return [rect.left(), rect.top(), rect.right(), rect.bottom()]

    def to_rect(self, box):
        xmin, ymin, xmax, ymax = box

        return dlib.rectangle(left=xmin, top=ymin, right=xmax, bottom=ymax)

