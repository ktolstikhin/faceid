import numpy as np


def merge_boxes(boxes, thres=0.3):
    '''Merge overlapping bounding boxes using the non-maximum suppression
    algorithm.

    Args:
        boxes (list): A list of bounding boxes [xmin, ymin, xmax, ymax].
        thres (float or None): A bounding box overlap threshold.
            Defaults to 0.3.

    Returns:
        A list of merged bounding boxes.
    '''

    if not boxes:
        return []

    if thres is None or thres <= 0:
        return boxes

    boxes = np.array(boxes, dtype='float')
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort the bounding boxes by
    # their bottom-right y-coordinates:
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(y2)

    while len(indices):
        # Grab the last index in the index list and add the index value to the
        # list of picked indexes:
        last = len(indices) - 1
        i = indices[last]

        pick.append(i)

        # Find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box:
        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])

        # Compute the width and the height of the bounding box:
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of the overlap:
        overlap = (w * h) / area[indices[:last]]

        # Remove bounding boxes which overlap the current box more than the
        # given threshold:
        indices = np.delete(
            indices,
            np.concatenate((
                [last],
                np.where(overlap > thres)[0]
            ))
        )

    return boxes[pick].astype('int').tolist()

