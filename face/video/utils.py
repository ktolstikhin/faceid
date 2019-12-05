import json

from .settings import VideoDeviceSettings
from .stream import VideoStream


def create_stream(conf_file, configure=False):

    with open(conf_file) as f:
        cfg = json.load(f)

    if configure:
        apply_device_settings(cfg)

    path = cfg['path']
    size = tuple(cfg['resolution'])

    return VideoStream(path, size)


def apply_device_settings(cfg):
    s = cfg.get('settings')

    if s is not None:
        vds = VideoDeviceSettings(cfg['path'])
        vds.exposure_manual()
        vds.set(s)

