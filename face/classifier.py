from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from .utils.logger import init_logger
from .model import build_model


class FaceClassifier:

    MODEL_PARAMS = None
    UNKNOWN_FACE_LABEL = 'Unknown_Face'

    def __init__(self, model_path=None, log=None):
        self.model = self.load(model_path)
        self.log = log or init_logger('faceid')

    def get_faces(self, path):
        self.log.info('Load face vectors from', path)
        vecs, names = [], []

        for face_dir in Path(path).glob('*'):

            if not face_dir.is_dir():
                continue

            npy_files = list(face_dir.glob('*.npy'))

            if not npy_files:
                self.log.warning('No face vectors found in', face_dir)
                continue

            for f in npy_files:
                vecs.append(np.load(f))
                names.append(face_dir.name)

        if not vecs:
            raise FileNotFoundError('No face vectors found.')

        return np.array(vecs), np.array(names)

    def train(self, face_db, test_size=0.2, average='macro'):
        X, y = self.get_faces(face_db)
        self.log.info('Split data, fit and score a model...')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        metrics = self.score(y_test, y_pred, average)

        self.log.info('Done. Average test metrics:', metrics)
        self.log.info('Train a final model using the whole data set...')

        probas = self.model.predict_proba(X_test)
        self.threshold = self.find_best_threshold(probas, y_test)

        self.model.fit(X, y)
        self.log.info('Done. Best prediction threshold:', self.threshold)

    def find_best_threshold(self, probas, y_true):
        thresholds = np.arange(0.1, 1.0, 0.01)
        corrs = []

        for thres in thresholds:
            y_pred = self.proba_to_labels(probas, thres)
            corr_coef = matthews_corrcoef(y_true, y_pred)
            corrs.append(corr_coef)

        best_i = np.argmax(corrs)

        return thresholds[best_i]

    def proba_to_label(self, probas, threshold=None):
        indices = np.argmax(probas, axis=1)
        labels = self.model.labels_.take(indices)

        max_probas = np.max(probas, axis=1)
        thres = threshold or self.threshold
        labels[max_probas < thres] = self.UNKNOWN_FACE_LABEL

        return labels

    def predict(self, face_vecs, proba=False, threshold=None):
        probas = self.model.predict_probas(face_vecs)
        labels = self.proba_to_label(probas, threshold)

        if not proba:
            return labels

        return [{'label': l, 'proba': p} for l, p in zip(labels, max_probas)]

    def score(self, y_test, y_pred, average='macro'):
        prec, recall, f1_score, _ = precision_recall_fscore_support(
            y_test, y_pred, average=average)

        metrics = {
            'average': average,
            'precision': '{prec:.5f}',
            'recall': '{recall:.5f}',
            'f1-score': '{f1_score:.5f}'
        }

        return metrics

    def test(self, face_db, average='macro'):
        X_test, y_test = self.get_faces(face_db)
        y_pred = self.predict(X_test)

        return self.score(y_test, y_pred, average)

    def save(self, path):
        self.log.info('Save a face recognizer model to', path)
        joblib.dump(self.model, path)

    def load(self, path=None):

        if path is None:
            return build_model(self.MODEL_PARAMS)

        self.log.info('Load a face recognizer model from', path)

        return joblib.load(path)

