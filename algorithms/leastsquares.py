import numpy as np
import math

from point import Point


def solve(points, args):
    visibility = math.inf
    if args.visibility is not None:
        visibility = float(args.visibility)

    sigma = 1
    if args.sigma is not None:
        sigma = args.sigma

    locations = []
    for point in points:
        if point.typ == "S":
            locations.append(tuple(point.coords))
        else:
            anchors = [p for p in points if p.dist(point) < visibility and p.typ == "S"]

            if len(anchors) < 3:
                print(f"Point {point} has only {len(anchors)} anchors. Cannot determine position.")
                locations.append((0 for __ in point))
                continue

            distances = [a.dist_noisy(point, sigma) for a in anchors]

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
