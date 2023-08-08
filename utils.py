"""utils.py
Various utilities to be used in algorithms."""


def collect_anchors_and_distances(point, points, visibility, sigma, min_anchors=3):
    """Find anchors in range and generate randomized distances to them"""
    anchors = [p for p in points if p.dist(point) < visibility and p.typ == "S"]
    if len(anchors) < min_anchors:
        raise ValueError("Not enough anchors")
    distances = [a.dist_noisy(point, sigma) for a in anchors]
    return anchors, distances
