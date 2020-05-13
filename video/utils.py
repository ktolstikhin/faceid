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


def apply_device_settings(cfg, reset=False):
    s = cfg.get('settings')

    if s is None:
        return

    vds = VideoDeviceSettings(cfg['path'])

    if reset:
        vds.reset_to_defaults()
    else:
        vds.exposure_manual()
        vds.set(s)

