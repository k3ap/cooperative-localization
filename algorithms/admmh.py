"""admmh.py
Implementation of the hybrid ADMM - Convex relaxation algorithm as proposed in
"Cooperative Localization is WSNs: A Hybrid Convex/Nonconvex Solution" by
N. Piovesan and T. Erseghe

Example usage:
`python main.py -f samples/standard/sample1.csv -a admmh -v 0.5 -s 0.05 -j 80`

Note that the algorithm "switches" between two distinct versions when the first
version finds a good-enough base solution. When (and whether) this switch occurs
is controlled by the TAU_C constant below.
This value is set sensibly for a 2D dataset in a unit square; if your dataset
includes coordinates of larger values, the parameter may need to be set higher.
"""


import numpy as np
import scipy.optimize

from network import Network, NetworkNode
from utils import random_vector


# Most of these parameters are set to the recommended
# values in the article
LAMMAX = 1000
EPS_C = 0.005
ZETA_C = 0.05
THETA_C = 0.98
DELTA_C = 1.01
TAU_C = 0.003


class HybridNode(NetworkNode):

    def __init__(self, point):
        super().__init__(point)
        self.switched = False

        if self.typ == "A":
            self.x = np.matrix(np.zeros((self.dim,))).T
        else:
            self.x = np.matrix(self.coords).T

        self.c = EPS_C
        self.prev_primal_gap = 0

    def postinit(self):
        """Post-initialization of edge values."""
        for edge in self.edges.values():
            if edge.typ == "S":
                edge.x = np.matrix(edge.pt.coords).T
            else:
                edge.x = np.matrix(np.zeros((self.dim,))).T

            edge.lam1 = np.matrix(np.zeros((self.dim,))).T
            edge.lam2 = np.matrix(np.zeros((self.dim,))).T
            edge.c = EPS_C

    def handle(self, msg, sender):
        # There are three types of messages:
        # (1, msg, sender) is the z_{1,i,j} value,
        # (2, msg, sender) is the z_{2,i,j} value,
        # and (3, msg, sender) is parameter c.
        num, msg = msg
        if num == 1:
            self.edges[sender].m1r = msg
        elif num == 2:
            self.edges[sender].m2r = msg
        else:
            self.edges[sender].c = msg

    def build_messages(self):
        for edge in self.edges.values():
            m1 = (self.x - edge.x) + edge.lam1 / self.c
            m2 = (self.x + edge.x) + edge.lam2 / self.c
            edge.m1s = m1
            edge.m2s = m2
            edge.send((1, m1))
            edge.send((2, m2))
            # we send message type 3 elsewhere

    def func(self, x):
        """The distance weight function at this node."""
        s = 0
        for edge in self.edges.values():
            if edge.typ == "S":
                argn = np.linalg.norm(x - np.array(edge.pt.coords))
                c = 2 * self.c + 1
            else:
                argn = np.linalg.norm(x - edge.y.T)
                c = 2 * self.c

            if self.switched or edge.dist < argn:
                s += (argn - edge.dist) * (argn - edge.dist) / c

        return 0.5 * float((x-self.y.T)*(x-self.y.T).T) + s * 0.5 / len(self.edges)

    def funcgrad(self, x):
        """The gradient of the distance weight function."""
        output = x - self.y.T
        for edge in self.edges.values():
            if edge.typ == "A":
                q = x - edge.y.T
                c = 2 * self.c + 1
            else:
                q = x - np.array(edge.pt.coords)
                c = 2 * self.c

            val = np.linalg.norm(q)

            if val > 0 and (self.switched or edge.dist < val):
                output += (val - edge.dist) * q / (c * len(self.edges) * val)

        return np.array(output).T[:,0]

    def iteration_begin(self):
        """First step of each (except the initial) iteration."""

        # Update the local y values

        if self.typ == "A":
            # We don't bother updating an anchor's y value, since we don't need it
            self.y = 0.5 / len(self.edges) * (
                sum(e.z1 + e.z2 - (e.lam1 + e.lam2)/self.c for e in self.edges.values())
            )

        for edge in self.edges.values():
            edge.y = 0.5 * (edge.z2 - edge.z1 - (edge.lam2 - edge.lam1) / self.c)

        # Update the local x values

        if self.typ == "S":
            for edge in self.edges.values():
                arg = edge.y - self.x
                argn = np.linalg.norm(arg)
                if argn > 0 and (self.switched or edge.dist < argn):
                    edge.x = self.x \
                        + (edge.dist + 2 * self.c * argn) \
                        * arg / (1 + 2 * self.c) / argn
                else:
                    edge.x = edge.y
        else:
            # this is an agent
            for edge in self.edges.values():
                arg = self.x - edge.y
                argn = np.linalg.norm(arg)
                if argn > 0 and (self.switched or edge.dist < argn):
                    edge.x = edge.y + arg * (argn - edge.dist) / argn / (1 + 2*self.c)
                else:
                    edge.x = edge.y

            x0 = self.x.T
            self.x = np.matrix(scipy.optimize.minimize(self.func, x0, jac=self.funcgrad).x).T

        # send out our new values
        # the initial iteration does this manually
        self.build_messages()

    def iteration_end(self, skip_lambda=False):
        """The final step of each iteration, including the initial iteration."""
        for edge in self.edges.values():
            edge.z1 = 0.5 * (edge.m1s - edge.m1r)
            edge.z2 = 0.5 * (edge.m2s + edge.m2r)

        if not skip_lambda:
            for edge in self.edges.values():
                newval = edge.lam1 + self.c * (self.x - edge.x - edge.z1)
                edge.lam1 = np.minimum(LAMMAX, np.maximum(-LAMMAX, newval))

                newval = edge.lam2 + self.c * (self.x + edge.x - edge.z2)
                edge.lam2 = np.minimum(LAMMAX, np.maximum(-LAMMAX, newval))

            primal_gap = 0
            for edge in self.edges.values():
                primal_gap = max(primal_gap, np.max(np.abs(self.x - edge.x - edge.z1)))
                primal_gap = max(primal_gap, np.max(np.abs(self.x + edge.x - edge.z2)))

            cmax = max(self.c, max(edge.c for edge in self.edges.values()))

            if self.switched:
                if primal_gap < self.prev_primal_gap * THETA_C:
                    self.c = cmax
                else:
                    self.c = cmax * DELTA_C
                self.broadcast((3, self.c))

            elif 0 < primal_gap < TAU_C or cmax > EPS_C:
                # in this situation, we switch to the other evaluation function
                self.switched = True
                self.c = ZETA_C
                self.broadcast((3, self.c))

            self.prev_primal_gap = primal_gap


def solve(points, args):
    network = Network(points, HybridNode, args)

    for pt in network.points:
        pt.postinit()

    # no need to do the first step of the iteration at the start
    for pt in network.points:
        pt.build_messages()

    for pt in network.points:
        pt.handle_messages()

    for pt in network.points:
        pt.iteration_end(skip_lambda=True)

    for iternum in range(args.iterations):
        for pt in network.points:
            pt.iteration_begin()

        for pt in network.points:
            pt.handle_messages()

        for pt in network.points:
            pt.iteration_end()

    return [tuple(float(x) for x in pt.x) for pt in network.points]


def animate(points, args):
    network = Network(points, HybridNode, args)

    for pt in network.points:
        pt.postinit()

    # no need to do the first step of the iteration at the start
    for pt in network.points:
        pt.build_messages()

    for pt in network.points:
        pt.handle_messages()

    for pt in network.points:
        pt.iteration_end(skip_lambda=True)

    for iternum in range(args.iterations):
        for pt in network.points:
            pt.iteration_begin()

        for pt in network.points:
            pt.handle_messages()

        for pt in network.points:
            pt.iteration_end()

        yield [tuple(float(x) for x in pt.x) for pt in network.points]

