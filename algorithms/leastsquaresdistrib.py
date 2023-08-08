"""leastsquaresdistrib.py
The least squares algorithm as a demonstration of NetworkPoint.
Should give the same results as `algorithms/leastsquares.py`.

Example usage:
`python main.py -f samples/sample1.csv -a leastsquaresdistrib -v 5 -s 0.05`
"""


import numpy as np

from distrib import NetworkPoint


class LSNetworkPoint(NetworkPoint):
    def get_location(self):
        """Calculate this point's estimated location."""
        if self.typ == "S":
            return tuple(self.coords)

        # Find all nearby anchors and the distance to them
        anchors = list(filter(lambda pt: pt.typ == "S", self.neighbours))
        distances = [d for pt, d in zip(self.neighbours, self.distances) if pt.typ == "S"]

        if len(anchors) < 3:
            print(f"Point {self} has too few anchors. Cannot determine position.")
            return tuple(0 for __ in point)
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
            return tuple(float(x) for x in loc)


def solve(points, args):

    # Convert the given Point list into LSNetworkPoint
    points = list(map(LSNetworkPoint, points))

    # We can only measure distances after we've filled in all neighbours
    for pt in points:
        pt.add_neighbours(points, args.visibility)

    for pt in points:
        pt.measure_distances(args.sigma)

    return list(map(LSNetworkPoint.get_location, points))
