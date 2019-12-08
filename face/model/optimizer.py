from sklearn.model_selection import GridSearchCV


def optimize_params(model, X, y, param_grid):
    search = GridSearchCV(model, param_grid, n_jobs=-1, refit=False)
    search.fit(X, y)

    return  search.best_params_

