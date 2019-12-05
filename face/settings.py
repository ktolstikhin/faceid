from pathlib import Path


root = Path(__file__).parent.joinpath('cfg')
model_conf_file = root.joinpath('models.json')
video_conf_files = list(root.joinpath('video').glob('*.json'))

clf_model_params = {
    'n_neighbors': 5,
    'algorithm': 'kd_tree',
    'n_jobs': -1
}

clf_thres_min = 0.3
clf_unknown_face_label = 'Unknown_Face'

