"""leastsquares.py
Contains a centralized implementation of the basic least squares algorithm.
This algorithm is non-cooperative, so every node needs to see at least three
anchors.
Example usage:
`python main.py -f samples/sample1.csv -a leastsquares -v 5 -s 0.05 -i`
Note that most samples have very few anchors relative to the number of points,
so you'll need to set a high visibility in order to use this algorithm.
`samples/sample1.csv` is an exception, as it has many anchors.
"""


import numpy as np
import math

from point import Point
from utils import collect_anchors_and_distances


def solve(points, args):
    # The calculated location estimation of all nodes
    locations = []

    for point in points:
        if point.typ == "S":
            # we know the exact position of this node
            locations.append(tuple(point.coords))
        else:
            try:
                anchors, distances = collect_anchors_and_distances(point, points, args.visibility, args.sigma)
            except ValueError:
                print(f"Point {point} has too few anchors. Cannot determine position.")
                locations.append(tuple(0 for __ in point))

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
