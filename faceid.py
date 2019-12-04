#!/usr/bin/python3
import os
import sys
import json
import time

import click
import numpy as np
from queue import Queue
from PIL import Image, ImageFile

from face import FaceWatcher, FaceRecognizer, settings
from face.vision import VisionTaskHandler
from face.utils.logger import init_logger


log = init_logger('faceid')
ImageFile.LOAD_TRUNCATED_IMAGES = True


@click.group()
def faceid():
    '''The application CLI.
    '''


@faceid.command()
@click.option('-t', '--task-handlers', type=int, default=1, show_default=True,
              help='A number of vision task handler threads.')
@click.option('-b', '--batch-size', type=int, default=32, show_default=True,
              help='A size of vision task batches.')
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

        for cfg_file in settings.video_conf_files:
            w = FaceWatcher(task_queue, cfg_file, show, log)
            w.start()
            watchers.append(w)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        # First, stop watchers, then stop handlers. The order does matter.

        for w in watchers:
            w.join()

        for h in handlers:
            h.join()


@faceid.group()
def model():
    '''A face recognizer model.
    '''


@model.command()
@click.option('-p', '--path', required=True, help='A path to a face DB.')
@click.option('-t', '--test-size', type=float, default=0.2, show_default=True,
              help='A size of a test part of the training data.')
@click.option('-o', '--output', help='A path to the output model.')
def train(path, test_size, output):
    '''Train a face recognizer model.
    '''
    log.info(f'Train a face recognition model on the face DB {path}')

    recognizer = FaceRecognizer(log)
    recognizer.clf.train(path, test_size)

    if output is None:
        output = os.path.join(path, 'face_clf.pkl')

    recognizer.clf.save(output)

    model_file = os.path.splitext(output)[0]
    score_file = f'{model_file}.json'

    with open(score_file, 'w') as f:
        score_json = json.dumps(recognizer.clf.model.score, indent=2)
        f.write(score_json)


@model.command()
@click.option('-p', '--path', required=True, help='A path to a test face DB.')
@click.option('-o', '--output', help='A path to output test metrics (JSON).')
def test(path, output):
    '''Test a face recognizer model.
    '''
    log.info(f'Test a face recognition model on the face DB {path}')

    recognizer = FaceRecognizer(log)
    score = recognizer.clf.test(path)

    score_json = json.dumps(score['macro avg'], indent=2)
    log.info(f'Done. Test score:\n{score_json}')

    if output is not None:

        with open(output, 'w') as f:
            score_json = json.dumps(score, indent=2)
            f.write(score_json)


@faceid.group()
def db():
    '''A face DB management.
    '''


@db.command()
@click.option('-p', '--path', required=True, help='A path to a face DB.')
@click.option('-f', '--force', is_flag=True,
              help='Force encoding all found images.')
def init(path, force):
    '''Initialize a face DB.
    '''
    recognizer = FaceRecognizer(log)

    input_files = [os.path.join(path, f) for f in os.listdir(path)]
    dirnames = [f for f in input_files if os.path.isdir(f)]
    n = len(dirnames)

    log.info(f'Inspecting the face DB {path}')
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
        dets = recognizer.detector.detect(img)

        if not dets:
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

