import numpy as np
import random
from math import sqrt

from distrib import NetworkPoint


class CRANetworkPoint(NetworkPoint):

    def __init__(self, point, spans):
        super().__init__(point)
        if self.typ == "S":
            self.x = np.matrix(self.coords).T
        else:
            self.x = np.matrix([
                random.uniform(mini-1, maxi+1) for mini, maxi in spans
            ]).T
        self.spans = spans
        self.xs = {}

    def process_signals(self):
        if self.typ == "S":
            self.message_queue.clear()
            return

        while len(self.message_queue) > 0:
            msg, sender = self.message_queue.popleft()
            self.xs[sender] = msg

    def update(self, lipscitz):
        z = np.matrix([
            random.uniform(mini-1, maxi+1) for mini, maxi in self.spans
        ]).T
        prev = np.matrix(z)
        l = 0
        while l == 0 or l < 500 and np.linalg.norm(z-prev) > 1e-3:
            l += 1
            w = z + (l-2)/(l+1)*(z - prev)
            df = 0.5 * w * len(self.neighbours)
            for i, pt in enumerate(self.neighbours):
                n = w - self.xs[pt]
                norm = np.linalg.norm(n)
                if norm > self.distances[i]:
                    n *= self.distances[i] / norm
                n += self.xs[pt]
                df -= 0.5*n
                if pt.typ == "S":
                    n += w - n

            prev = np.matrix(z)
            z = w - df / lipscitz

        self.x = z
        self.broadcast(self.x)

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

    points = list(map(lambda x: CRANetworkPoint(x, spans), points))

    for pt in points:
        pt.add_neighbours(points, args.visibility)

    for pt in points:
        pt.measure_distances(args.sigma)

    for pt in points:
        pt.broadcast(pt.x)

    for pt in points:
        pt.process_signals()

    maxdegree = 0
    maxanchors = 0
    for pt in points:
        maxdegree = max(maxdegree, len(pt.neighbours))
        maxanchors = max(maxanchors, pt.num_anchor_neighbours())

    lipschitz = 2 * maxdegree + maxanchors

    for iternum in range(args.iterations):
        # choose random node and update it
        chosen = random.choice(points)
        while chosen.typ == "S":
            chosen = random.choice(points)

        chosen.update(lipschitz)
        for pt in chosen.neighbours:
            pt.process_signals()

    return [tuple(map(float, pt.x)) for pt in points]
