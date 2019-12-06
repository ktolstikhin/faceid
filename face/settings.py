from pathlib import Path


root = Path(__file__).parent.joinpath('cfg')
model_conf_file = root.joinpath('models.json')
video_conf_files = list(root.joinpath('video').glob('*.json'))

# Default parameters for face model:
clf_model_params = {
    'n_neighbors': 5,
    'algorithm': 'kd_tree',
    'n_jobs': -1
}

# Parameter grid for model optimization:
clf_model_param_grid = {
    'n_neighbors': range(1, 6),
    'weights': ('uniform', 'distance'),
    'algorithm': ('ball_tree', 'kd_tree'),
}

clf_unknown_face_label = 'Unknown_Face'

