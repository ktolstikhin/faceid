def box_center(bbox):
    xmin, ymin, xmax, ymax = bbox
    x = xmin + (xmax - xmin) / 2
    y = ymin + (ymax - ymin) / 2

    return int(x), int(y)


def overlap(c1min, c1max, c2min, c2max):
    return c1max >= c2min and c2max >= c1min


def box_in_roi(bbox, roi):
    x1min, y1min, x1max, y1max = bbox
    x2min, y2min, x2max, y2max = roi

    return (overlap(x1min, x1max, x2min, x2max)
            and overlap(y1min, y1max, y2min, y2max))


def point_in_roi(pt, roi):
    x, y = pt
    xmin, ymin, xmax, ymax = roi

    return xmin < x < xmax and ymin < y < ymax

