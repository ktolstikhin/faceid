import os
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, classification_report

from .model import builder, optimizer
from .utils.logger import init_logger
from . import settings


class FaceClassifier:

    def __init__(self, model_path=None, log=None):
        self.log = log or init_logger('faceid')
        self.model = self.load(model_path)

    def get_faces(self, path):
        self.log.info(f'Load face vectors from {path}')
        vecs, names = [], []

        for face_dir in Path(path).glob('*'):

            if not face_dir.is_dir():
                continue

            npy_files = list(face_dir.glob('*.npy'))

            if not npy_files:
                self.log.warning(f'No face vectors found in {face_dir}')
                continue

            for f in npy_files:
                vecs.append(np.load(f))
                names.append(face_dir.name)

        if not vecs:
            raise FileNotFoundError('No face vectors found.')

        return np.array(vecs), np.array(names)

    def train(self, face_db, test_size=0.2, optimize_params=False):
        self.log.info('Split data, then fit and score the face model...')
        X, y = self.get_faces(face_db)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=True)
        except TypeError:
            self.log.warning('Stratified split failed, use a regular one.')
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size)

        self.log.info(f'Data: train {X_train.shape}, test {X_test.shape}')

        if optimize_params:
            self.log.info('Start optimization of model parameters...')
            params = optimizer.optimize(
                self.model, X, y, settings.clf_model_param_grid)
            self.log.info(f'Best params: {params}')
            params.update({'n_jobs': -1})
            self.model.set_params(**params)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.model.score = self.score(y_test, y_pred)

        score_json = json.dumps(self.model.score['macro avg'], indent=2)
        self.log.info(f'Test model score:\n{score_json}')
        self.log.info('Train a final model...')

        probas = self.model.predict_proba(X_test)
        self.model.threshold = self.find_best_threshold(probas, y_test)

        self.model.fit(X, y)
        self.log.info(f'Done. Best threshold: {self.model.threshold:.2f}')

    def find_best_threshold(self, probas, y_true):
        thresholds = np.arange(settings.clf_min_thres, 1.0, 0.01)[::-1]
        corrs = []

        for thres in thresholds:
            y_pred = self.proba_to_label(probas, thres)
            corr_coef = matthews_corrcoef(y_true, y_pred)
            corrs.append(corr_coef)

        best_i = np.argmax(corrs)

        return thresholds[best_i]

    def proba_to_label(self, probas, threshold=None):
        indices = np.argmax(probas, axis=1)
        labels = self.model.classes_.take(indices)
        max_probas = np.max(probas, axis=1)
        thres = threshold or self.model.threshold
        labels[max_probas < thres] = settings.clf_unknown_face_label

        return labels

    def predict(self, face_vecs, threshold=None, proba=False):
        probas = self.model.predict_proba(face_vecs)
        labels = self.proba_to_label(probas, threshold)

        if not proba:
            return labels

        max_probas = np.max(probas, axis=1)

        return [{'label': l, 'proba': p} for l, p in zip(labels, max_probas)]

    def score(self, y_test, y_pred):
        return classification_report(y_test, y_pred, output_dict=True)

    def test(self, face_db):
        X_test, y_test = self.get_faces(face_db)
        y_pred = self.predict(X_test)

        return self.score(y_test, y_pred)

    def save(self, path):
        self.log.info(f'Save a face recognizer model to {path}')
        joblib.dump(self.model, path)

    def load(self, path=None):

        if path is None:
            self.log.info('Build a new face recognizer model...')
            model = builder.build_model(settings.clf_model_params)
        else:
            self.log.info(f'Load a face recognizer model from {path}')
            model = joblib.load(path)

        return model

