"""admm.py
An implementation of Algorithm 1 from "A Distributed and Maximum-Likelihood
Sensor Network Localization Algorithm Based Upon a Nonconvex Problem Formulation"
by T. Erseghe
We use scipy.optimize.minimize for the optimization problem at each step.

Example usage:
`python main.py -f samples/sample2.csv -a admm -v 5 -s 0.05 -j 40`
Note that this algorithm is considerably slower than many others.
"""


from math import sqrt
import scipy.optimize
import numpy as np

from network import Network, NetworkNode


# Algorithm constants, as recommended in the article
EPS = 1/2
ZETA = 1/10
SQRTEPS = sqrt(EPS)
SQRTZETA = sqrt(ZETA)
DELTA_C = 1.01
SIGMA = 1


class ADMMNetworkNode(NetworkNode):

    def __init__(self, point):
        super().__init__(point)
        self.c = 2

    # The algorithm's first iterative step differs slightly from the others,
    # so we make special methods for it.

    def first_iteration_start(self):
        if self.typ == "S":
            self.x = np.matrix(self.coords).T
        else:
            self.x = np.matrix(np.zeros( (self.dim, 1) ))

        for edge in self.edges_ordered():
            edge.x = np.matrix(np.zeros( (self.dim, 1) ))
            edge.lam1 = np.matrix(np.zeros( (self.dim, 1) ))
            edge.lam2 = np.matrix(np.zeros( (self.dim, 1) ))

        self.build_messages()

    def build_messages(self):
        """Build and send messages"""
        for edge in self.edges_ordered():
            m1 = SQRTEPS * (self.x - edge.x) + edge.lam1 / self.c
            m2 = SQRTZETA * (self.x + edge.x) + edge.lam2 / self.c
            edge.m1s = m1
            edge.m2s = m2
            edge.send((1, m1))
            edge.send((2, m2))

    def handle(self, msg, sender):
        num, msg = msg
        edge = self.edges[sender]
        if num == 1:
            edge.m1r = msg
        else:
            edge.m2r = msg

    def update_z(self):
        for edge in self.edges_ordered():
            edge.z1 = 0.5 * (edge.m1s - edge.m1r)
            edge.z2 = 0.5 * (edge.m2s + edge.m2r)


    def update_y(self):
        sw1 = sum(edge.z1 - edge.lam1 / self.c for edge in self.edges.values())
        sw2 = sum(edge.z2 - edge.lam2 / self.c for edge in self.edges.values())

        self.y = sw2 / (2 * SQRTZETA * len(self.edges))
        self.y += sw1 / (2 * SQRTEPS * len(self.edges))

        for edge in self.edges.values():
            edge.y = (EPS - ZETA) / (4 * EPS * ZETA * len(self.edges))\
                * (SQRTEPS * sw1 + SQRTZETA * sw2)
            edge.y += (SQRTZETA * (edge.z2 - edge.lam2 / self.c)
                       - SQRTEPS * (edge.z1 - edge.lam1 / self.c))
            edge.y += (ZETA-EPS)*(ZETA-EPS)\
                / (4 * (ZETA+EPS) * ZETA * EPS * len(self.edges))\
                * (SQRTZETA * sw2 - SQRTEPS * sw1)

    def first_iteration_end(self):
        self.update_z()
        self.c *= DELTA_C
        self.update_y()

    def func(self, vals):
        """Evaluate current position"""

        def sqr(x):
            """Square the argument"""
            return x*x

        if self.typ == "S":
            # If this is an anchor, the parameter has self.dim dimensions fewer
            # (the value of self.x is fixed), so we handle it differently
            s = 0
            for i, edge in enumerate(self.edges_ordered()):
                s += sqr(edge.dist - np.linalg.norm(self.x - vals[i*self.dim:(i+1)*self.dim]))
            s /= SIGMA

            s2 = (ZETA+EPS) * len(self.edges) * sqr(np.linalg.norm(self.x - self.y))
            for i, edge in enumerate(self.edges_ordered()):
                s2 += (ZETA+EPS) * float(sqr(np.linalg.norm( vals[i*self.dim:(i+1)*self.dim] - edge.y.T )))
                s2 += 2 * (ZETA-EPS) * float(np.dot( vals[i*self.dim:(i+1)*self.dim] - edge.y.T, (self.x - self.y) ))

            s += 0.5 * self.c * s2
            return s
        else:
            # this is an agent
            s = 0
            for i, edge in enumerate(self.edges_ordered()):
                s += sqr(edge.dist - np.linalg.norm(vals[0:self.dim] - vals[(i+1)*self.dim:(i+2)*self.dim]))
            s /= SIGMA

            s2 = (ZETA+EPS) * len(self.edges) * sqr(np.linalg.norm(vals[0:self.dim] - self.y.T))
            for i, edge in enumerate(self.edges_ordered()):
                s2 += (ZETA+EPS)*float(sqr(np.linalg.norm( vals[(i+1)*self.dim:(i+2)*self.dim] - edge.y.T )))
                s2 += 2 * (ZETA-EPS) * float(np.dot( vals[(i+1)*self.dim:(i+2)*self.dim] - edge.y.T, (vals[0:self.dim] - self.y.T).T ))

            s += 0.5 * self.c * s2
            return s

    def iteration_start(self):
        if self.typ == "S":
            # we constrain the problem so that this node's position is exact
            x0 = np.zeros((self.dim * len(self.edges),))
            for i, edge in enumerate(self.edges_ordered()):
                x0[i*self.dim:(1+i)*self.dim] = edge.x.T

            xs = scipy.optimize.minimize(self.func, x0).x

            for i, edge in enumerate(self.edges_ordered()):
                edge.x = np.matrix(xs[i*self.dim:(1+i)*self.dim]).T
        else:
            x0 = np.zeros((self.dim * (1 + len(self.edges)),))
            x0[0:self.dim] = self.x.T
            for i, edge in enumerate(self.edges_ordered()):
                x0[(1+i)*self.dim:(2+i)*self.dim] = edge.x.T

            xs = scipy.optimize.minimize(self.func, x0).x

            self.x = np.matrix(xs[0:self.dim]).T
            for i, edge in enumerate(self.edges_ordered()):
                edge.x = np.matrix(xs[(1+i)*self.dim:(2+i)*self.dim]).T

        self.build_messages()

    def iteration_end(self):
        self.update_z()

        for edge in self.edges.values():
            edge.lam1 += self.c * (SQRTEPS * (self.x - edge.x) - edge.z1)
            edge.lam2 += self.c * (SQRTZETA * (self.x + edge.x) - edge.z2)

        self.c *= DELTA_C
        self.update_y()


def solve(points, args):
    network = Network(points, ADMMNetworkNode, args)

    for pt in network.points:
        pt.first_iteration_start()

    for pt in network.points:
        pt.handle_messages()

    for pt in network.points:
        pt.first_iteration_end()

    for iternum in range(args.iterations):
        for pt in network.points:
            pt.iteration_start()

        for pt in network.points:
            pt.handle_messages()

        for pt in network.points:
            pt.iteration_end()

    return [tuple(float(c) for c in pt.x) for pt in network.points]
