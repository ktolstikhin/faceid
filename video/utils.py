import json

from .stream import VideoStream
from .process import VideoStreamProcess
from .settings import VideoDeviceSettings


def create_stream(conf_file, configure=False, multiprocessing=False):

    with open(conf_file) as f:
        cfg = json.load(f)

    if configure:
        apply_device_settings(cfg)

    path = cfg['path']
    size = tuple(cfg['resolution'])

    if not multiprocessing:
        stream = VideoStream(path, size)
    else:
        stream = VideoStreamProcess(path, size)

    return stream


def apply_device_settings(cfg, reset=False):
    s = cfg.get('settings')

    if s is not None:
        vds = VideoDeviceSettings(cfg['path'])

        if reset:
            vds.reset_to_defaults()
        else:
            vds.exposure_manual()
            vds.set(s)

