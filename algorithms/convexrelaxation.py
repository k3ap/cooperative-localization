"""convexrelaxation.py
The parallel version of the convex relaxation algorithm, as described in
"Simple and Fast Convex Relaxation Method for Cooperative Localization in Sensor
Networks Using Range Measurements" by C. Soares, J. Xavier, and J. Gomes

Example usage:
`python main.py -f samples/standard/sample2.csv -a convexrelaxation -v 4 -s 0.05 -j 100`
"""

import numpy as np
import random
from math import sqrt

from network import NetworkNode, Network
from utils import random_vector


class CRNetworkNode(NetworkNode):

    def __init__(self, point, spans):
        super().__init__(point)

        if self.typ == "S":
            self.x = np.matrix(self.coords).T
        else:
            # Randomize x at the start
            self.x = random_vector(spans)

        self.prev = self.x

    def begin_iteration(self, iternum):
        """Initiate an iteration at this node."""
        self.w = self.x + (iternum - 2) / (iternum + 1) * (self.x - self.prev)
        self.broadcast(self.w)

    def handle(self, msg, sender):
        self.edges[sender].w = msg

    def end_iteration(self, lipschitz):
        """The final step of the iteration, and the bulk of the algorithm."""

        if self.typ == "S":
            return

        dg = len(self.edges) * self.w - sum(e.w for e in self.edges.values())
        for edge in self.edges.values():

            # Compute the projection of w_i onto B(w_j, d_ij)
            n = self.w - edge.w
            norm = np.linalg.norm(n)
            if norm > edge.dist:
                n *= edge.dist / norm

            dg -= n

        dh = self.w * self.num_anchor_neighbours()
        for edge in self.edges.values():
            if edge.typ != "S":
                continue

            # compute the projection of w_i onto B(a_k, d_ik)
            a = np.matrix(edge.pt.coords).T  # this is an anchor
            n = self.w - a
            norm = np.linalg.norm(n)
            if norm > edge.dist:
                n *= edge.dist / norm
            n += a

            dh -= n

        self.prev = np.matrix(self.x)
        self.x = self.w - (dg + dh) / lipschitz

    def num_anchor_neighbours(self):
        """Get the number of anchors which are neighbouring this node"""
        num = 0
        for edge in self.edges.values():
            if edge.typ == "S":
                num += 1
        return num



def solve(points, args):
    # Calculate the bounds for the problem
    spans = [[c,c] for c in points[0]._coords]
    for pt in points:
        for i in range(len(pt._coords)):
            spans[i][0] = min(spans[i][0], pt._coords[i])
            spans[i][1] = max(spans[i][1], pt._coords[i])

    network = Network(points, CRNetworkNode, args, spans)

    # Calculate the lipschitz constant
    maxdegree = 0
    maxanchors = 0
    for pt in network.points:
        maxdegree = max(maxdegree, len(pt.edges))
        maxanchors = max(maxanchors, pt.num_anchor_neighbours())

    lipschitz = 2 * maxdegree + maxanchors

    # Update nodes
    for iternum in range(1, 1+args.iterations):
        for pt in network.points:
            pt.begin_iteration(iternum)

        for pt in network.points:
            pt.handle_messages()

        for pt in network.points:
            pt.end_iteration(lipschitz)

    return [tuple(map(float, pt.x)) for pt in network.points]


def animate(points, args):
    # Calculate the bounds for the problem
    spans = [[c,c] for c in points[0]._coords]
    for pt in points:
        for i in range(len(pt._coords)):
            spans[i][0] = min(spans[i][0], pt._coords[i])
            spans[i][1] = max(spans[i][1], pt._coords[i])

    network = Network(points, CRNetworkNode, args, spans)

    # Calculate the lipschitz constant
    maxdegree = 0
    maxanchors = 0
    for pt in network.points:
        maxdegree = max(maxdegree, len(pt.edges))
        maxanchors = max(maxanchors, pt.num_anchor_neighbours())

    lipschitz = 2 * maxdegree + maxanchors

    # Update nodes
    for iternum in range(1, 1+args.iterations):
        for pt in network.points:
            pt.begin_iteration(iternum)

        for pt in network.points:
            pt.handle_messages()

        for pt in network.points:
            pt.end_iteration(lipschitz)

        yield [tuple(map(float, pt.x)) for pt in network.points]
