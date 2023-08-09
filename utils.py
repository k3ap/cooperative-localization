"""utils.py
Various utilities to be used in algorithms."""


import numpy as np
import random
import string


def collect_anchors_and_distances(point, points, visibility, sigma, min_anchors=3):
    """Find anchors in range and generate randomized distances to them"""
    anchors = [p for p in points if p.typ == "S" and p._dist(point) < visibility]
    if len(anchors) < min_anchors:
        raise ValueError("Not enough anchors")
    distances = [a.dist_noisy(point, sigma) for a in anchors]
    return anchors, distances


def random_vector(spans):
    """Return a random vector close to the problem bounds."""
    return np.matrix([
        random.uniform(mini-1, maxi+1) for mini, maxi in spans
    ]).T


GENERATED_UIDS = set()

def generate_uid(length=8):
    """Generate a unique identifier of specified length."""
    def gen_string(l):
        s = ""
        for __ in range(l):
            s += random.choice(string.ascii_letters)
        return s

    while (s := gen_string(length)) in GENERATED_UIDS:
        pass

    GENERATED_UIDS.add(s)
    return s
