import numpy as np
from scipy.spatial import distance

from .keeper import FaceTargetKeeper
from .target import FaceTarget
from .utils import box_center


class FaceTracker:

    TARGET_LOST_FRAMES = 10

    def __init__(self, img_size):
        self.img_size = img_size
        self.target_keeper = FaceTargetKeeper()
        self.targets = {}
        self.lost_frames = {}

    def bound_size(self, box):
        xmin, ymin, xmax, ymax = box
        w, h = self.img_size

        return max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)

    def update_target(self, target_id, face):

        try:
            target = self.targets[target_id]

            if target.label != face['label']:
                self.target_keeper.remove(target)
                target.label = face['label']
                target.proba = face['proba']
                target.box = self.bound_size(face['box'])
                self.target_keeper.add(target)

            self.lost_frames[target_id] = 0
        except KeyError:
            pass

    def add_target(self, face):
        target = FaceTarget(face['label'], face['proba'], face['box'])
        self.target_keeper.add(target)
        self.targets[target.id] = target
        self.lost_frames[target.id] = 0

    def remove_target(self, target_id):

        try:
            target = self.targets[target_id]
            self.target_keeper.remove(target)
            del self.targets[target_id]
            del self.lost_frames[target_id]
        except KeyError:
            pass

    def get_targets(self):
        return self.targets.values()

    def update(self, faces):
        # If the input list of detected faces is empty, increment lost frame
        # counter for all of the tracked targets. If number of frames without a
        # tracked target has exceeded a certain threshold, remove this target
        # from the tracked ones.
        if not faces:
            lost_targets = []

            for target_id in self.targets:
                self.lost_frames[target_id] += 1

                if self.lost_frames[target_id] > self.TARGET_LOST_FRAMES:
                    lost_targets.append(target_id)

            for tid in lost_targets:
                self.remove_target(tid)

            return self

        # If there are no tracked targets, add all the detected faces to the
        # tracked targets.
        if not self.targets:

            for face in faces:
                self.add_target(face)

            return self

        # Perform a match between centroids of bounding boxes of the detected
        # faces and tracked targets based on the euclidean distance between
        # them.
        target_ids, target_centroids = [], []

        for tid, target in self.targets.items():
            target_ids.append(tid)
            center = box_center(target.box)
            target_centroids.append(center)

        target_centroids = np.array(target_centroids)
        input_centroids = [box_center(face['box']) for face in faces]
        input_centroids = np.array(input_centroids)

        dist = distance.cdist(target_centroids, input_centroids)

        # Sort distances between centroids. Here, "rows" is index array of
        # tracked targets, and "cols" is index array of closest detected faces.
        rows = dist.min(axis=1).argsort()
        cols = dist.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):

            if row in used_rows or col in used_cols:
                continue

            # Update the tracked target:
            target_id = target_ids[row]
            self.update_target(target_id, faces[col])

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(dist.shape[0])).difference(used_rows)
        unused_cols = set(range(dist.shape[1])).difference(used_cols)

        # If number of tracked targets is greater than the number of detected
        # faces, update lost frame counters of tracked targets which have not
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
                self.add_target(faces[col])

        return self

