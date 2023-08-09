"""convexrelaxation-old.py
The parallel version of the convex relaxation algorithm, as described in
"Simple and Fast Convex Relaxation Method for Cooperative Localization in Sensor
Networks Using Range Measurements" by C. Soares, J. Xavier, and J. Gomes

Example usage:
`python main.py -f samples/sample2.csv -a convexrelaxation-old -v 4 -s 0.05 -j 300`
"""

import numpy as np
import random
from math import sqrt

from distrib import NetworkPoint
from utils import random_vector


class CRNetworkPoint(NetworkPoint):

    def __init__(self, point, spans):
        super().__init__(point)

        # Randomize x at the start
        self.x = random_vector(spans)
        self.prev = self.x

        # We need to keep track of neighbouring nodes' estimated positions
        self.ws = {}

    def begin_iteration(self, iternum):
        """Initiate an iteration at this node."""
        self.w = self.x + (iternum - 2) / (iternum + 1) * (self.x - self.prev)
        self.broadcast(self.w)

    def process_signals(self):
        """The second step of the algorithm is to send and receive the position
        estimates w."""
        while len(self.message_queue) > 0:
            msg, sender = self.message_queue.popleft()
            self.ws[sender] = msg

    def end_iteration(self, lipschitz):
        """The final step of the iteration, and the bulk of the algorithm."""

        dg = len(self.neighbours) * self.w - sum(self.ws.values())
        for i, pt in enumerate(self.neighbours):

            # Compute the projection of self.w onto B(self.ws[pt], self.distances[i])
            n = self.w - self.ws[pt]
            norm = np.linalg.norm(n)
            if norm > self.distances[i]:
                n *= self.distances[i] / norm

            dg -= n

        dh = sum(self.w for k in self.ws.keys() if k.typ == "S")
        for i, pt in enumerate(self.neighbours):
            if pt.typ != "S":
                continue

            # compute the projection of self.w onto B(pt, self.distances[i])
            a = np.matrix(pt.coords).T  # pt is an anchor
            n = self.w - a
            norm = np.linalg.norm(n)
            if norm > self.distances[i]:
                n *= self.distances[i] / norm
            n += a

            dh -= n

        self.prev = np.matrix(self.x)
        self.x = self.w - (dg + dh) / lipschitz

    def num_anchor_neighbours(self):
        """Get the number of anchors which are neighbouring this node"""
        num = 0
        for pt in self.neighbours:
            if pt.typ == "S":
                num += 1
        return num



def solve(points, args):
    # Calculate the bounds for the problem
    spans = [[c,c] for c in points[0]._coords]
    for pt in points:
        for i in range(len(pt._coords)):
            spans[i][0] = min(spans[i][0], pt._coords[i])
            spans[i][1] = max(spans[i][1], pt._coords[i])

    points = list(map(lambda x: CRNetworkPoint(x, spans), points))

    for pt in points:
        pt.add_neighbours(points, args.visibility)

    for pt in points:
        pt.measure_distances(args.sigma)

    # Calculate the lipschitz constant
    maxdegree = 0
    maxanchors = 0
    for pt in points:
        maxdegree = max(maxdegree, len(pt.neighbours))
        maxanchors = max(maxanchors, pt.num_anchor_neighbours())

    lipschitz = 2 * maxdegree + maxanchors

    # Update nodes
    for iternum in range(1, 1+args.iterations):
        for pt in points:
            pt.begin_iteration(iternum)

        for pt in points:
            pt.process_signals()

        for pt in points:
            pt.end_iteration(lipschitz)

    return [tuple(map(float, pt.x)) for pt in points]
