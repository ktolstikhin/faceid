from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


def build_model(params=None):
    svc = SVC(probability=True)
    model = OneVsRestClassifier(svc, n_jobs=-1)

    if params is not None:
        model.set_params(**params)

    return model

