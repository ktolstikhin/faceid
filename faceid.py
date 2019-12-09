#!/usr/bin/python3
import os
import sys
import json
import time
from queue import Queue

import click
import numpy as np
from PIL import Image

from face import settings
from face.watcher import FaceWatcher
from face.classifier import FaceClassifier
from face.recognizer import FaceRecognizer
from face.utils.logger import init_logger
from face.tracker import FaceTargetKeeper
from face.vision import VisionTaskHandler
from face.video.frame import FrameBuffer
from face.video.utils import create_stream, apply_device_settings


log = init_logger('faceid')


@click.group()
def faceid():
    '''The application CLI.
    '''


@faceid.command()
@click.option('-t', '--task-handlers', type=int, default=1, show_default=True,
              help='A number of vision task handler threads.')
@click.option('-b', '--batch-size', type=int, default=32, show_default=True,
              help='A size of a vision task batch.')
@click.option('-s', '--show', is_flag=True, help='Show tracked faces.')
def run(task_handlers, batch_size, show):
    '''Start watching faces.
    '''
    try:
        task_queue = Queue()
        handlers, watchers = [], []
        log.info(f'Start {task_handlers} vision task handler(s)...')

        for _ in range(task_handlers):
            h = VisionTaskHandler(task_queue, batch_size, log)
            h.start()
            handlers.append(h)

        watcher_num = len(settings.video_conf_files)
        log.info(f'Start {watcher_num} face watcher(s)...')

        for conf_file in settings.video_conf_files:
            video_stream = create_stream(conf_file)
            log.info(f'Start video stream from {video_stream.path}')
            w = FaceWatcher(task_queue, video_stream, show, log)
            w.start()
            watchers.append(w)

        target_keeper = FaceTargetKeeper()
        frame_buffer = FrameBuffer(log=log) if show else None

        while True:
            tracked = target_keeper.get()
            faces = {label: len(targets) for label, targets in tracked.items()}
            results = ', '.join(f'{num} {lab}' for lab, num in faces.items())

            if not results:
                results = 'No faces found.'

            log.info(f'Tracking: {results}')

            if frame_buffer is not None:
                frame_buffer.show()

    except KeyboardInterrupt:
        pass
    finally:
        # First, join watchers, then handlers. The order does matter.
        threads = watchers + handlers

        for t in threads:
            t.join()


@faceid.group()
def video():
    '''Manage video devices.
    '''


@video.command()
@click.option('-r', '--reset', is_flag=True,
              help='Reset device settings to defaults.')
def config(reset):
    '''Configure video devices.
    '''

    for cfg_file in settings.video_conf_files:

        with open(cfg_file) as f:
            cfg = json.load(f)

        log.info(f'Initialize video device {cfg["path"]}')
        apply_device_settings(cfg, reset)


@video.command()
@click.option('-c', '--conf-json', required=True,
              help='A path to the video device configuration.')
@click.option('-o', '--output', help='A path to the output images.')
def live(conf_json, output):
    '''Start a live video stream.
    '''

    if output is not None:
        os.makedirs(output, exist_ok=True)

    frame_buffer = FrameBuffer(output, log)
    video_stream = create_stream(conf_json)
    log.info(f'Start video stream from {video_stream.path}')

    try:

        while True:
            frame = video_stream.read()

            if frame is None:
                log.warning(f'Failed to read from {video_stream.path}')
                continue

            frame_buffer.add(frame, title=video_stream.path)
            frame_buffer.show()

    except KeyboardInterrupt:
        pass


@faceid.group()
def model():
    '''Train and test a face recognizer model.
    '''


@model.command()
@click.option('-f', '--facedb', required=True,
              help='A path to the face database.')
@click.option('-t', '--test-size', type=float, default=0.2, show_default=True,
              help='A size of a test part of the training data.')
@click.option('-o', '--output', help='A path to the output model.')
@click.option('--optimize', is_flag=True, help='Optimize model parameters.')
def train(facedb, test_size, output, optimize):
    '''Train a face recognizer model.
    '''
    log.info(f'Train a face recognizer on the face database {facedb}')

    clf = FaceClassifier(log=log)
    clf.train(facedb, test_size, optimize)

    if output is None:
        output = os.path.join(facedb, 'face_clf.pkl')

    clf.save(output)
    model_file = os.path.splitext(output)[0]
    score_file = f'{model_file}.json'

    with open(score_file, 'w') as f:
        score_json = json.dumps(clf.model.score, indent=2)
        f.write(score_json)


@model.command()
@click.option('-f', '--facedb', required=True,
              help='A path to the face database.')
@click.option('-m', '--model', required=True,
              help='A path to the face recognizer model.')
@click.option('-o', '--output', help='A path to output test metrics (json).')
def test(facedb, model, output):
    '''Test a face recognizer model.
    '''
    log.info(f'Test a face recognizer on the face database {facedb}')

    clf = FaceClassifier(model, log)
    score = clf.test(facedb)

    score_json = json.dumps(score['macro avg'], indent=2)
    log.info(f'Done. Test score:\n{score_json}')

    if output is not None:
        log.info(f'Save test metrics to {output}')

        with open(output, 'w') as f:
            score_json = json.dumps(score, indent=2)
            f.write(score_json)


@faceid.group()
def db():
    '''Manage a face database.
    '''


@db.command()
@click.option('-f', '--facedb', required=True,
              help='A path to the face database.')
@click.option('--force', is_flag=True, help='Force encoding all found images.')
def init(facedb, force):
    '''Initialize a face database.
    '''
    recognizer = FaceRecognizer(log)

    input_files = [os.path.join(facedb, f) for f in os.listdir(facedb)]
    dirnames = [f for f in input_files if os.path.isdir(f)]
    n = len(dirnames)

    log.info(f'Inspecting the face database {facedb}')
    images = []

    for i, dirname in enumerate(dirnames, start=1):
        log.info(f'[{i}/{n}] {dirname}')

        files = [os.path.join(dirname, f) for f in os.listdir(dirname)]
        img_files = [
            f for f in files if f.endswith('.jpg') and 'aligned' not in f]

        if not img_files:
            log.warning('No images found.')
            continue

        if force:
            images.extend(img_files)
            continue

        # Find face images which don't have embeddings (.npy files):
        npy_files = {
            os.path.splitext(f)[0] for f in files if f.endswith('.npy')
        }

        for f in img_files:
            name = os.path.splitext(f)[0]

            if name not in npy_files:
                images.append(f)

    if not images:
        return

    n = len(images)
    log.info(f'Found {n} images for processing.')

    for i, img_file in enumerate(images, start=1):
        log.info(f'[{i}/{n}] Process image {img_file}')
        img = np.array(Image.open(img_file))
        dets = recognizer.detector.detect([img])[0]

        if not len(dets):
            log.warning('No faces found. Ignore image.')
            continue

        face = dets[0]
        face_img = recognizer.aligner.align(img, face)

        img_dir = os.path.dirname(img_file)
        basename = os.path.basename(img_file)
        name, ext = os.path.splitext(basename)
        face_img_file = os.path.join(img_dir, f'{name}_aligned{ext}')

        log.info(f'Save aligned face to {face_img_file}')
        Image.fromarray(face_img).save(face_img_file)

        face_emb = recognizer.encoder.encode([face_img])[0]

        face_emb_file = os.path.join(img_dir, f'{name}.npy')
        log.info(f'Save face embedding to {face_emb_file}')
        np.save(face_emb_file, face_emb)


if __name__ == '__main__':
    faceid()

