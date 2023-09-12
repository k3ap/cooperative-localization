"""lsangle.py
A least squares-based algorithm which only uses angle measurements.
"""


from math import sin, cos, degrees, sqrt
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
                w = np.array([cos(edge.angle), sin(edge.angle)])
                diff = (xj - xi) / np.linalg.norm(xj - xi)
                s += np.sum((w - diff) * (w - diff))
        return s

    def grad(x):
        res = np.zeros(x.shape)
        for pt in network.points:
            if pt.typ == "S":
                continue

            for edge in pt.edges.values():
                xi = get(x, pt)
                xj = get(x, edge._dest)
                w = np.array([cos(edge.angle), sin(edge.angle)])
                d = (xi[0] - xj[0])*(xi[0] - xj[0]) + (xi[1] - xj[1])*(xi[1] - xj[1])
                d *= sqrt(d)
                res[indicies[pt]] += (xi[1] - xj[1]) * ((xi[1] - xj[1]) * cos(edge.angle) + (xj[0] - xi[0]) * sin(edge.angle)) / d
                res[indicies[pt]+1] += (xi[0] - xj[0]) * ((xj[1] - xi[1]) * cos(edge.angle) + (xi[0] - xj[0]) * sin(edge.angle)) / d

        return res*2

    x0 = np.random.rand(idx)
    x = scipy.optimize.minimize(func, x0, jac=grad).x

    ret = []
    for pt in network.points:
        if pt.typ == "S":
            ret.append(pt.coords)
        else:
            ret.append(tuple(x[indicies[pt]:indicies[pt]+pt.dim]))

    return ret
