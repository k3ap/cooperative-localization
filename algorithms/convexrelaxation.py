import numpy as np
import random
from math import sqrt

from distrib import NetworkPoint


class CRNetworkPoint(NetworkPoint):

    def __init__(self, point, spans):
        super().__init__(point)
        self.x = np.matrix([
            random.uniform(mini-1, maxi+1) for mini, maxi in spans
        ]).T
        self.prev = self.x
        self.ws = {}

    def begin_iteration(self, iternum):
        self.w = self.x + (iternum - 2) / (iternum + 1) * (self.x - self.prev)
        self.broadcast(self.w)

    def process_signals(self):
        while len(self.message_queue) > 0:
            msg, sender = self.message_queue.popleft()
            self.ws[sender] = msg

    def end_iteration(self, lipschitz):
        dg = len(self.neighbours) * self.w - sum(self.ws.values())
        for i, pt in enumerate(self.neighbours):
            n = self.w - self.ws[pt]
            norm = np.linalg.norm(n)
            if norm > self.distances[i]:
                n *= self.distances[i] / norm
            dg -= n

        dh = sum(self.w for k in self.ws.keys() if k.typ == "S")
        for i, pt in enumerate(self.neighbours):
            if pt.typ != "S":
                continue

            n = np.matrix(self.w)
            a = np.matrix(pt.coords).T  # pt is an anchor
            n -= a
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
    spans = [[c,c] for c in points[0].coords]

    for pt in points:
        for i in range(len(pt.coords)):
            spans[i][0] = min(spans[i][0], pt.coords[i])
            spans[i][1] = max(spans[i][1], pt.coords[i])

    points = list(map(lambda x: CRNetworkPoint(x, spans), points))

    for pt in points:
        pt.add_neighbours(points, args.visibility)

    for pt in points:
        pt.measure_distances(args.sigma)

    maxdegree = 0
    maxanchors = 0
    for pt in points:
        maxdegree = max(maxdegree, len(pt.neighbours))
        maxanchors = max(maxanchors, pt.num_anchor_neighbours())

    lipschitz = 2 * maxdegree + maxanchors

    for iternum in range(1, 1+args.iterations):
        for pt in points:
            pt.begin_iteration(iternum)

        for pt in points:
            pt.process_signals()

        for pt in points:
            pt.end_iteration(lipschitz)

    return [tuple(map(float, pt.x)) for pt in points]
