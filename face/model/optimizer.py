from sklearn.model_selection import GridSearchCV


def optimize(model, X, y, params):
    search = GridSearchCV(model, params, n_jobs=-1)
    search.fit(X, y)

    return  search.best_params_

