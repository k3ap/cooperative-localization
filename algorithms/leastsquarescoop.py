"""leastsquarescoop.py
Cooperative version of the least squares algorithm.
"""


import numpy as np
import scipy.optimize

from network import Network, NetworkNode


def sqr(x):
    return x*x


def solve(points, args):

    network = Network(points, NetworkNode, args)

    num_agents = 0
    for pt in network.points:
        if pt.typ == "A":
            num_agents += 1

    def get_positions(locs, dim):
        idx = 0
        positions = {}
        for pt in network.points:
            if pt.typ == "S":
                positions[pt] = np.matrix(pt.coords).T
            else:
                positions[pt] = np.matrix(locs[idx:idx+dim]).T
                idx += dim
        return positions

    def func(locs):
        """Function we're trying to minimize; the total square error"""
        s = 0
        positions = get_positions(locs, network.points[0].dim)
        for pt in network.points:
            for edge in pt.edges.values():
                s += sqr(edge.dist - np.linalg.norm(positions[pt]-positions[edge._dest]))
        return s

    def grad(locs):
        dim = network.points[0].dim
        s = np.zeros((dim * num_agents,))
        positions = get_positions(locs, dim)
        idx = 0
        for pt in network.points:
            if pt.typ == "S":
                continue

            for edge in pt.edges.values():
                n = np.linalg.norm(positions[pt] - positions[edge._dest])
                for k in range(dim):
                    s[idx+k] += (edge.dist - n) * (positions[pt][k,0] - positions[edge._dest][k,0]) / n

            idx += dim

        return -2 * s


    x0 = np.random.rand(points[0].dim * num_agents)
    locs = scipy.optimize.minimize(func, x0, jac=grad).x

    positions = get_positions(locs, network.points[0].dim)
    return [tuple(float(x) for x in positions[pt]) for pt in network.points]
