import numpy as np

from distrib import NetworkPoint


class LSNetworkPoint(NetworkPoint):
    def get_distance(self):
        if self.typ == "S":
            return tuple(self.coords)

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
    points = list(map(LSNetworkPoint, points))
    for pt in points:
        pt.add_neighbours(points, args.visibility)

    for pt in points:
        pt.measure_distances(args.sigma)

    return list(map(LSNetworkPoint.get_distance, points))
