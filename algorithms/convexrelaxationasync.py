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

from network import NetworkNode, Network
from utils import random_vector


class CRANetworkNode(NetworkNode):

    def __init__(self, point, spans):
        super().__init__(point)

        if self.typ == "S":
            # We'll be reading this variable as an intermediate and final
            # result, so we'll just set it to the precise value
            self.x = np.matrix(self.coords).T
        else:
            # If we don't know the precise value, we set it to random in the
            # start
            self.x = random_vector(spans)

        # We'll need to keep track of this for initialization during updating
        self.spans = spans

    def handle(self, msg, sender):
        if self.typ == "S":
            return

        self.edges[sender].x = msg

    def update(self, lipscitz):

        if self.typ == "S":
            return

        # Randomize z at the start
        z = random_vector(self.spans)
        prev = np.matrix(z)

        # The cutoff for maximum iterations has been set to 500, but this
        # amount is normally not reached in practice (at least on the included
        # samples)
        l = 0
        while l == 0 or l < 500 and np.linalg.norm(z-prev) > 1e-3:
            l += 1

            w = z + (l-2)/(l+1)*(z - prev)

            df = 0.5 * w * len(self.edges)

            for edge in self.edges.values():
                # calculate the projection of w onto B(x_j, d_ij)
                n = w - edge.x
                norm = np.linalg.norm(n)
                if norm > edge.dist:
                    n *= edge.dist / norm
                n += edge.x

                df -= 0.5*n

                if edge.typ == "S":
                    # the second sum actually computes the exact
                    # same thing, but for anchors only
                    df += w - n

            prev = np.matrix(z)
            z = w - df / lipscitz

        self.x = z
        self.broadcast(self.x)

    def num_anchor_neighbours(self):
        """Get the number of anchors which are neighbouring this node"""
        num = 0
        for edge in self.edges.values():
            if edge.typ == "S":
                num += 1
        return num


def solve(points, args):
    # Calculate the bounds for the problem
    spans = [[c,c] for c in points[0].coords]
    for pt in points:
        for i in range(len(pt._coords)):
            spans[i][0] = min(spans[i][0], pt._coords[i])
            spans[i][1] = max(spans[i][1], pt._coords[i])

    network = Network(points, CRANetworkNode, args, spans)

    # Nodes need to be aware of each other's position estimated
    # at the very beginning
    for pt in network.points:
        pt.broadcast(pt.x)

    for pt in network.points:
        pt.handle_messages()

    # Calculate the lipschitz constant
    maxdegree = 0
    maxanchors = 0
    for pt in network.points:
        maxdegree = max(maxdegree, len(pt.edges))
        maxanchors = max(maxanchors, pt.num_anchor_neighbours())

    lipschitz = 2 * maxdegree + maxanchors

    # Update nodes
    for iternum in range(args.iterations):
        # choose random node and update it
        chosen = random.choice(network.points)
        while chosen.typ == "S":
            chosen = random.choice(network.points)

        chosen.update(lipschitz)
        for edge in chosen.edges.values():
            # This cheating is ok, because we don't learn any secret
            # properties from doing it
            edge._dest.handle_messages()

    return [tuple(map(float, pt.x)) for pt in network.points]


def animate(points, args):
    # Calculate the bounds for the problem
    spans = [[c,c] for c in points[0].coords]
    for pt in points:
        for i in range(len(pt._coords)):
            spans[i][0] = min(spans[i][0], pt._coords[i])
            spans[i][1] = max(spans[i][1], pt._coords[i])

    network = Network(points, CRANetworkNode, args, spans)

    # Nodes need to be aware of each other's position estimated
    # at the very beginning
    for pt in network.points:
        pt.broadcast(pt.x)

    for pt in network.points:
        pt.handle_messages()

    # Calculate the lipschitz constant
    maxdegree = 0
    maxanchors = 0
    for pt in network.points:
        maxdegree = max(maxdegree, len(pt.edges))
        maxanchors = max(maxanchors, pt.num_anchor_neighbours())

    lipschitz = 2 * maxdegree + maxanchors

    # Update nodes
    for iternum in range(args.iterations):
        # choose random node and update it
        chosen = random.choice(network.points)
        while chosen.typ == "S":
            chosen = random.choice(network.points)

        chosen.update(lipschitz)
        for edge in chosen.edges.values():
            # This cheating is ok, because we don't learn any secret
            # properties from doing it
            edge._dest.handle_messages()

        yield [tuple(map(float, pt.x)) for pt in network.points]
