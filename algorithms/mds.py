"""mds.py
An implementation of the multidimensional scaling algorithm as proposed in
"Sensor Positioning in Wireless Ad-hoc Sensor Networks Using Multidimensional Scaling"
by X. Ji, H. Zha
"""

import numpy as np

from network import NetworkNode, Network


def solve(points, args):
    network = Network(points, NetworkNode, args)

    indicies = dict((pt, idx) for idx, pt in enumerate(network.points))
    n = len(points)

    V = np.zeros((n, n))
    anchors = []
    for pt in network.points:
        if pt.typ == "S":
            anchors.append(pt)

        for edge in pt.edges.values():
            V[indicies[pt], indicies[pt]] += 1
            V[indicies[pt], indicies[edge._dest]] += -1

    Vinv = np.linalg.inv(V + np.ones((n,n)))

    X = np.random.rand(n, 2) - 0.5 * np.ones((n, 2))

    for iternum in range(args.iterations):
        B = np.zeros((n, n))

        for pt in network.points:
            for edge in pt.edges.values():
                dist = np.linalg.norm( X[indicies[pt],:] - X[indicies[edge._dest],:] )
                val = edge.dist / dist
                B[indicies[pt], indicies[pt]] += val
                B[indicies[pt], indicies[edge._dest]] -= val

        X = Vinv @ B @ X

    # We will need to transform the predicted locations into the original
    # coordinate matrix
    # For testing purposes, we write the predicted locations to a file, which
    # can be displayed by visualize.py
    with open("mds-predicted-locations.csv", "w") as f:
        for pt in network.points:
            f.write(f"{X[indicies[pt],0]},{X[indicies[pt],1]},{pt.typ}\n")

    # when transforming, we need to take into account that
    # the generated coordinates can be rotated, flipped and misaligned
    # We therefore fix in place three anchors, using their known position to
    # estimate the transformation matrix

    def collinear(pt1, pt2, pt3):
        """Check whether three points are collinear."""
        x1 = pt1.coords[0] - pt2.coords[0]
        y1 = pt1.coords[1] - pt2.coords[1]
        x2 = pt1.coords[0] - pt3.coords[0]
        y2 = pt1.coords[1] - pt3.coords[1]
        return abs(x1 * y2 - x2 * y1) < 1e-3

    # We have to pick three non-collinear points
    while collinear(anchors[0], anchors[1], anchors[-1]):
        anchors.pop()

    # Build the transformation matrix such that it solves
    # [x2 - x1, y2 - y1]^T = Q [s2 - s1, t2 - t1]^T
    # [x3 - x1, y3 - y1]^T = Q [s3 - s1, t3 - t1]^T
    # In the ideal scenario, the determinant of Q should be +1 or -1
    Q = np.zeros((2,2))
    x1 = X[indicies[anchors[0]],0]
    y1 = X[indicies[anchors[0]],1]
    x2 = X[indicies[anchors[1]],0]
    y2 = X[indicies[anchors[1]],1]
    x3 = X[indicies[anchors[-1]],0]
    y3 = X[indicies[anchors[-1]],1]
    t1 = anchors[0].coords[0]
    s1 = anchors[0].coords[1]
    t2 = anchors[1].coords[0]
    s2 = anchors[1].coords[1]
    t3 = anchors[-1].coords[0]
    s3 = anchors[-1].coords[1]

    n1 = (y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2))
    n2 = (x3 * (y1-y2) + x1 * (y2 - y3) + x2 * (y3 - y1))

    Q[0,0] = (y1 * (t2 - t3) + y2 * (t3 - t1) + y3 * (t1 - t2)) / n1
    Q[0,1] = (t3 * (x2 - x1) + t2 * (x1 - x3) + t1*(x3 - x2)) / n2
    Q[1,0] = (y1 * (s2 - s3) + y2 * (s3 - s1) + y3 * (s1 - s2)) / n1
    Q[1,1] = (s3 * (x2 - x1) + s2 * (x1 - x3) + s1 * (x3 - x2)) / n2

    # The translation
    v1 = X[indicies[anchors[0]],:]
    w1 = np.array(anchors[0].coords)

    return [tuple(Q @ (X[idx,:] - v1) + w1) for idx in range(len(points))]
