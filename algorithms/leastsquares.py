import numpy as np
import math

from point import Point
from utils import collect_anchors_and_distances


def solve(points, args):
    locations = []
    for point in points:
        if point.typ == "S":
            locations.append(tuple(point.coords))
        else:
            try:
                anchors, distances = collect_anchors_and_distances(point, points, args.visibility, args.sigma)
            except ValueError:
                print(f"Point {point} has too few anchors. Cannot determine position.")
                locations.append((0 for __ in point))

            else:

                A = np.matrix([
                    [x - y for x, y in zip(anchors[0], a)]
                    for a in anchors[1:]
                ])
                b = np.matrix([
                    anchors[0].abssq() - a.abssq() - distances[0]*distances[0] + d*d
                    for a, d in zip(anchors[1:], distances[1:])
                ]).T

                loc = 0.5 * np.linalg.inv(A.T * A) * A.T * b
                locations.append(tuple(float(x) for x in loc))

    return locations
