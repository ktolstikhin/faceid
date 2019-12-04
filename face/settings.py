from pathlib import Path


root = Path(__file__).parent.joinpath('cfg')
model_conf_file = root.joinpath('models.json')

clf_model_params = {
    'estimator__gamma': 'auto',
    'estimator__kernel': 'rbf',
    'estimator__probability': True,
    'estimator__verbose': False,
    'n_jobs': -1
}
clf_thres_min = 0.3
clf_unknown_face_label = 'Unknown_Face'

