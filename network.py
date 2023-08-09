"""network.py
An abstract graph representation of a network, meant to replace distrib.py.
"""


from point import Point
import random
from collections import deque

from utils import generate_uid


class NetworkEdge:
    def __init__(self, source, dest):
        self._source = source
        self._dest = dest
        super().__setattr__("props", {})
        self.props["typ"] = self._dest.typ

        if self._dest.typ == "S":
            self.props["pt"] = self._dest

    def __getattr__(self, name):
        if name not in self.props:
            raise ValueError(f"{name} is not an edge property; did you forget to transmit it?")

        return self.props[name]

    def __setattr__(self, name, value):
        if name[0] == "_":
            return super().__setattr__(name, value)
        self.props[name] = value

    def _send(self, msg):
        """Send a message along this edge"""
        self._dest._receive(msg, self._source._uid)


class NetworkNode(Point):
    def __init__(self, point):
        super().__init__(*point._coords, typ=point.typ)
        self.edges = {}
        self._uid = generate_uid()
        self.message_queue = deque()

    def __str__(self):
        return "NP(" + ", ".join(map(str, self._coords)) + ")"

    def _receive(self, message, sender):
        self.message_queue.append((message, sender))

    def handle_messages(self):
        while len(self.message_queue) > 0:
            msg, sender = self.message_queue.popleft()
            self.handle(msg, sender)

    def broadcast(self, msg):
        for edge in self.edges:
            edge._send(msg)

    def handle(self, msg, sender):
        pass


class Network:
    def __init__(self, points, point_cls, args):
        self.points = list(map(point_cls, points))

        self._add_neighbours(args.visibility)
        self._measure_distances(args.sigma)

    def _add_neighbours(self, visibility):
        for pt1 in self.points:
            for pt2 in self.points:
                if pt1._distsq(pt2) < visibility*visibility:
                    pt1.edges[pt2._uid] = NetworkEdge(pt1, pt2)

    def _measure_distances(self, sigma):
        """Measure synchronized noisy distances between nodes."""
        for pt1 in self.points:
            for edge in pt1.edges.values():
                edge.dist = pt1.dist_noisy(edge._dest, sigma)
