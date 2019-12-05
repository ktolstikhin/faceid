from sklearn.neighbors import KNeighborsClassifier


def build_model(params=None):
    model = KNeighborsClassifier()

    if params is not None:
        model.set_params(**params)

    return model

