"""convexrelaxationasync.py
The asynchronous version of the convex relaxation algorithm, as described in
"Simple and Fast Convex Relaxation Method for Cooperative Localization in Sensor
Networks Using Range Measurements" by C. Soares, J. Xavier, and J. Gomes

Example usage:
`python main.py -f samples/sample2.csv -a convexrelaxationasync -v 4 -s 0.05 -j 400`
"""


import numpy as np
import random
from math import sqrt

from distrib import NetworkPoint


class CRANetworkPoint(NetworkPoint):

    def __init__(self, point, spans):
        super().__init__(point)

        if self.typ == "S":
            # We'll be reading this variable as an intermediate and final
            # result, so we'll just set it to the precise value
            self.x = np.matrix(self.coords).T
        else:
            # If we don't know the precise value, we set it to random in the
            # start
            self.x = np.matrix([
                random.uniform(mini-1, maxi+1) for mini, maxi in spans
            ]).T

        # We'll need to keep track of this for initialization during updating
        self.spans = spans

        # The estimated positions of neighbouring nodes
        self.xs = {}

    def process_signals(self):
        """Process signals.
        All signals are assumed to be a copy of a node's self.x attribute."""
        if self.typ == "S":
            # No need to keep track of neighbours as an anchor
            self.message_queue.clear()
            return

        while len(self.message_queue) > 0:
            msg, sender = self.message_queue.popleft()
            self.xs[sender] = msg

    def update(self, lipscitz):

        # Randomize z at the start
        z = np.matrix([
            random.uniform(mini-1, maxi+1) for mini, maxi in self.spans
        ]).T
        prev = np.matrix(z)

        # The cutoff for maximum iterations has been set to 500, but this
        # amount is normally not reached in practice (at least on the included
        # samples)
        l = 0
        while l == 0 or l < 500 and np.linalg.norm(z-prev) > 1e-3:
            l += 1

            w = z + (l-2)/(l+1)*(z - prev)

            df = 0.5 * w * len(self.neighbours)

            for i, pt in enumerate(self.neighbours):
                # calculate the projection of w onto B(self.xs[pt], self.distances[i])
                n = w - self.xs[pt]
                norm = np.linalg.norm(n)
                if norm > self.distances[i]:
                    n *= self.distances[i] / norm
                n += self.xs[pt]

                df -= 0.5*n

                if pt.typ == "S":
                    # the second sum actually computes the exact
                    # same thing, but for anchors only
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
    # Calculate the bounds for the problem
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

    # Nodes need to be aware of each other's position estimated
    # at the very beginning
    for pt in points:
        pt.broadcast(pt.x)

    for pt in points:
        pt.process_signals()

    # Calculate the lipschitz constant
    maxdegree = 0
    maxanchors = 0
    for pt in points:
        maxdegree = max(maxdegree, len(pt.neighbours))
        maxanchors = max(maxanchors, pt.num_anchor_neighbours())

    lipschitz = 2 * maxdegree + maxanchors

    # Update nodes
    for iternum in range(args.iterations):
        # choose random node and update it
        chosen = random.choice(points)
        while chosen.typ == "S":
            chosen = random.choice(points)

        chosen.update(lipschitz)
        for pt in chosen.neighbours:
            pt.process_signals()

    return [tuple(map(float, pt.x)) for pt in points]
