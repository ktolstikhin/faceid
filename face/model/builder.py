from sklearn.neighbors import KNeighborsClassifier


def build_model(params=None):
    model = KNeighborsClassifier(n_jobs=-1)

    if params is not None:
        model.set_params(**params)

    return model

