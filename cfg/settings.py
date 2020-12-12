from pathlib import Path

from PIL import ImageFile


root = Path(__file__).parent
model_conf_file = root.joinpath('models.json')
video_conf_files = list(root.joinpath('video').glob('*.json'))

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Default parameters for face model:
clf_model_params = {
    'n_neighbors': 5,
    'weights': 'uniform',
    'algorithm': 'kd_tree'
}

# Parameter grid for model optimization:
clf_model_param_grid = {
    'n_neighbors': range(1, 6),
    'weights': ('uniform', 'distance'),
    'algorithm': ('ball_tree', 'kd_tree'),
}

clf_unknown_face_label = 'Unknown_Face'

logger = 'face_id'

