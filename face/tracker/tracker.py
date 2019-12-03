import numpy as np
from scipy.spatial import distance

from .target import Target
from .utils.plane import box_center


class Tracker:

    TARGET_LOST_FRAMES = 10

    def __init__(self, img_size):
        self.img_size = img_size
        self.targets = {}
        self.lost_frames = {}

    def bound_size(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w, h = self.img_size

        return max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)

    def add_target(self, bbox):
        self.targets[target.id] = Target(bbox)
        self.lost_frames[target.id] = 0

    def remove_target(self, target_id):
        del self.targets[target_id]
        del self.lost_frames[target_id]

    def get_targets(self):
        return list(self.targets.values())

    def update(self, bbox_list):
        # If the input list of detected objects is empty, increment lost frame
        # counter for all of the tracked targets. If number of frames without a
        # tracked target has exceeded a certain threshold, remove this target
        # from the tracked ones.
        if not bbox_list:
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

            for bbox in bbox_list:
                self.add_target(bbox)

            return self

        # Perform a match between centroids of bounding boxes of the detected
        # objects and tracked targets based on the euclidean distance between
        # them.
        target_ids, target_centroids = [], []

        for tid, target in self.targets.items():
            target_ids.append(tid)
            target_centroids.append(target.center)

        input_centroids = [box_center(bbox) for bbox in bbox_list]

        target_centroids = np.array(target_centroids)
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

            # Update bounding box coordinates of the tracked target:
            target_id = target_ids[row]
            target = self.targets[target_id]
            target.bbox = self.bound_size(bbox_list[col])
            self.lost_frames[target_id] = 0

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
                self.add_target(bbox_list[col])

        return self

