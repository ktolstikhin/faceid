from functools import partial
from collections import defaultdict, deque, Counter

import numpy as np
from scipy.spatial import distance

from .target import Target
from .keeper import TargetKeeper
from .utils import box_center


class TargetTracker:

    TARGET_LOST_FRAMES = 10
    TARGET_LABEL_CANDIDATES = 10

    def __init__(self, img_size):
        self.img_size = img_size
        self.target_keeper = TargetKeeper()
        self.targets = {}
        self.lost_frames = {}
        label_list = partial(deque, maxlen=self.TARGET_LABEL_CANDIDATES)
        self.target_labels = defaultdict(label_list)

    def bound_size(self, box):
        xmin, ymin, xmax, ymax = box
        w, h = self.img_size

        return max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)

    def update_target(self, target_id, prediction):

        try:
            target = self.targets[target_id]
            target.proba = prediction['proba']
            target.box = self.bound_size(prediction['box'])

            new_label = prediction['label']
            target_labels = self.target_labels[target_id]
            target_labels.append(new_label)

            if target.label != new_label:
                counter = Counter(target_labels)
                most_common_label = counter.most_common()[0][0]

                if most_common_label == new_label:
                    self.target_keeper.remove(target)
                    target.label = new_label
                    self.target_keeper.add(target)

            self.lost_frames[target_id] = 0
        except KeyError:
            pass

    def add_target(self, prediction):
        target = Target(prediction['label'],
                        prediction['proba'],
                        prediction['box'])
        self.target_keeper.add(target)
        self.targets[target.id] = target
        self.lost_frames[target.id] = 0
        self.target_labels[target.id].append(prediction['label'])

    def remove_target(self, target_id):

        try:
            target = self.targets[target_id]
            self.target_keeper.remove(target)
            del self.targets[target_id]
            del self.lost_frames[target_id]
            del self.target_labels[target_id]
        except KeyError:
            pass

    def get_targets(self):
        return self.targets.values()

    def update(self, predictions):
        # If the input list of detected objects is empty, increment lost frame
        # counter for all of the tracked targets. If number of frames without a
        # tracked target has exceeded a certain threshold, remove this target
        # from the tracked ones.
        if not predictions:
            lost_targets = []

            for target_id in self.targets:
                self.lost_frames[target_id] += 1

                if self.lost_frames[target_id] > self.TARGET_LOST_FRAMES:
                    lost_targets.append(target_id)

            for tid in lost_targets:
                self.remove_target(tid)

            return self

        # If there are no tracked targets, add all the detected objects to the
        # tracked targets.
        if not self.targets:

            for p in predictions:
                self.add_target(p)

            return self

        # Perform a match between centroids of bounding boxes of the detected
        # objects and tracked targets based on the euclidean distance between
        # them.
        target_ids, target_centroids = [], []

        for tid, target in self.targets.items():
            target_ids.append(tid)
            center = box_center(target.box)
            target_centroids.append(center)

        target_centroids = np.array(target_centroids)
        input_centroids = [box_center(p['box']) for p in predictions]
        input_centroids = np.array(input_centroids)

        dist = distance.cdist(target_centroids, input_centroids)

        # Sort distances between centroids. Here, "rows" is index array of
        # tracked targets, and "cols" is index array of closest detected
        # objects.
        rows = dist.min(axis=1).argsort()
        cols = dist.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):

            if row in used_rows or col in used_cols:
                continue

            # Update the tracked target:
            target_id = target_ids[row]
            self.update_target(target_id, predictions[col])

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(dist.shape[0])).difference(used_rows)
        unused_cols = set(range(dist.shape[1])).difference(used_cols)

        # If number of tracked targets is greater than the number of detected
        # objects, update lost frame counters of tracked targets which have not
        # been found. Otherwise, add new targets to the tracked ones.
        if dist.shape[0] > dist.shape[1]:
            lost_targets = []

            for row in unused_rows:
                target_id = target_ids[row]
                self.lost_frames[target_id] += 1

                if self.lost_frames[target_id] > self.TARGET_LOST_FRAMES:
                    lost_targets.append(target_id)

            for tid in lost_targets:
                self.remove_target(tid)

        else:

            for col in unused_cols:
                self.add_target(predictions[col])

        return self

