"""lsangle.py
A least squares-based algorithm which takes angles into account.
"""


from math import sin, cos, degrees
import random
import numpy as np
import scipy.optimize

from network import NetworkNode, Network


def solve(points, args):
    network = Network(points, NetworkNode, args)
    network.measure_angles(args.sigma_angles)

    indicies = {}
    idx = 0
    for pt in network.points:
        if pt.typ == "A":
            indicies[pt] = idx
            idx += pt.dim

    def get(x, pt):
        if pt.typ == "S":
            return np.array(pt.coords)
        else:
            return x[indicies[pt]:indicies[pt]+pt.dim]

    def func(x):
        s = 0
        for pt in network.points:
            for edge in pt.edges.values():
                xi = get(x, pt)
                xj = get(x, edge._dest)
                y = np.array([cos(edge.angle), sin(edge.angle)]) * edge.dist
                s += np.sum((xj-xi-y)*(xj-xi-y))
        return s

    def grad(x):
        res = np.zeros(x.shape)
        for pt in network.points:
            if pt.typ == "S":
                continue

            for edge in pt.edges.values():
                xi = get(x, pt)
                xj = get(x, edge._dest)
                y = np.array([cos(edge.angle), sin(edge.angle)]) * edge.dist
                for k in range(pt.dim):
                    res[indicies[pt]+k] += xi[k] - xj[k] + y[k]

        return res*4

    x0 = np.zeros((idx,))
    x = scipy.optimize.minimize(func, x0, jac=grad).x

    ret = []
    for pt in network.points:
        if pt.typ == "S":
            ret.append(pt.coords)
        else:
            ret.append(tuple(x[indicies[pt]:indicies[pt]+pt.dim]))

    return ret
