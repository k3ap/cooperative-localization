"""leastsquaresnetworked.py
The least squared noncooperative algorithm as a demonstration of network.py.
"""


import numpy as np

from network import NetworkNode, Network


class LSNode(NetworkNode):
    def solve(self):
        if self.typ == "S":
            return tuple(self.coords)

        # Find all nearby anchors and the distance to them
        anchors = list(filter(lambda e: e.typ == "S", self.edges.values()))

        if len(anchors) < 3:
            print(f"Point {self} has too few anchors. Cannot determine position.")
            return tuple(0 for __ in self)
        else:
            A = np.matrix([
                [x - y for x, y in zip(anchors[0].pt, a.pt)]
                for a in anchors[1:]
            ])
            b = np.matrix([
                anchors[0].pt.abssq() - a.pt.abssq() \
                - anchors[0].dist*anchors[0].dist + a.dist * a.dist
                for a in anchors[1:]
            ]).T

            loc = 0.5 * np.linalg.inv(A.T * A) * A.T * b
            return tuple(float(x) for x in loc)


def solve(points, args):
    network = Network(points, LSNode, args)
    return list(map(LSNode.solve, network.points))
