"""network.py
An abstract graph representation of a network, meant to replace distrib.py.

To use this, subclass NetworkNode, make a network with your new subclass in the
solve(...) function and call the desired functions.
Check algorithms/leastsquaresnetworked.py for an easy example and
algorithms/admmh.py for a more advanced example.
"""


from point import Point
import random
from collections import deque
from math import atan2, pi
import heapq

from utils import generate_uid


class DisconnectedGraphError(Exception):
    pass


class NetworkEdge:
    """An edge in the network graph. Can be used to send information to
    the neighbour node, which is then saved in its copy of the edge."""

    def __init__(self, source, dest):

        # Source and destination for this node.
        # You should avoid using these fields, except for debug purposes.
        self._source = source
        self._dest = dest

        # Received information is saved as edge properties
        super().__setattr__("props", {})
        self.props["typ"] = self._dest.typ

        # To make implementation easier, every anchor property is accessible
        # with edge.pt
        if self._dest.typ == "S":
            self.props["pt"] = self._dest

    def __getattr__(self, name):
        if name not in self.props:
            s = f"{name} is not an edge property; did you forget to transmit it?"
            raise ValueError(s)

        return self.props[name]

    def __setattr__(self, name, value):
        if name[0] == "_":
            return super().__setattr__(name, value)
        self.props[name] = value

    def send(self, msg):
        """Send a message along this edge"""
        self._dest._receive(msg, self._source._uid)

    def __str__(self):
        return f"NewtorkEdge({self._source}, {self._dest})"

    def __lt__(self, o):
        """Less-than. Required in building MST."""
        return str(self) < str(o)


class NetworkNode(Point):
    """A single node in the network."""

    def __init__(self, point):
        super().__init__(*point._coords, typ=point.typ)

        # The edges of this node. It is assumed no new edges will be added
        # during the algorithm's run
        self.edges = {}

        # Some algorithms require a consistent ordering of edges, which is
        # provided with edges_ordered()
        self._order = []

        # Messages are queued before they're handled.
        # To handle every queued message, call self.handle_messages()
        self.message_queue = deque()

        # The unique identifier of this node.
        # These identifiers are used as keys in self.edges
        # They're not secret, but there's often little need for them
        self._uid = generate_uid()

    def __str__(self):
        return "NP(" + ", ".join(map(str, self._coords)) + ")"

    def __repr__(self):
        return str(self)

    def _receive(self, message, sender):
        self.message_queue.append((message, sender))

    def handle_messages(self):
        while len(self.message_queue) > 0:
            msg, sender = self.message_queue.popleft()
            self.handle(msg, sender)

    def broadcast(self, msg):
        """Send a message to all neighbouring nodes."""
        for edge in self.edges.values():
            edge.send(msg)

    def handle(self, msg, sender):
        """Hadle a sent message. Override this method in your subclass."""
        pass

    def edges_ordered(self):
        """An ordered view of all edges."""
        for uid in self._order:
            yield self.edges[uid]


class Network:
    """The network graph."""

    def __init__(self, points, point_cls, args, *node_init_args, check_disconnect=True):
        self._point_cls = point_cls
        self._args = args
        self._node_init_args = node_init_args
        self.points = list(map(lambda p: point_cls(p, *node_init_args), points))

        self._add_neighbours(args.visibility)
        self._measure_distances(args.sigma)

        if check_disconnect and not self._check_connectivity():
            raise DisconnectedGraphError("Graph is disconnected.")

    def _check_connectivity(self):
        """Return True if the network is connected."""
        seen = set()

        def dfs(pt):
            seen.add(pt)
            for edge in pt.edges.values():
                if edge._dest not in seen:
                    dfs(edge._dest)

        dfs(self.points[0])
        for pt in self.points:
            if pt not in seen:
                return False

        return True

    def _add_neighbours(self, visibility):
        for pt1 in self.points:
            for pt2 in self.points:
                if pt1 is pt2:
                    continue

                if pt1._distsq(pt2) < visibility*visibility:
                    pt1.edges[pt2._uid] = NetworkEdge(pt1, pt2)

            pt1._order = list(pt1.edges)

    def _measure_distances(self, sigma):
        """Measure synchronized noisy distances between nodes."""
        for pt1 in self.points:
            for edge in pt1.edges.values():
                edge.dist = pt1.dist_noisy(edge._dest, sigma)
                edge._dest.edges[pt1._uid].dist = edge.dist

    def measure_angles(self, sigma):
        """Measure synchronized noisy angles between nodes. Only works on 2D problems, and should
        be called from the algorithm if required."""
        assert self.points[0].dim == 2

        for pt in self.points:
            for edge in pt.edges.values():
                diff = tuple(c1 - c2 for c1, c2 in zip(edge._dest._coords, pt._coords))
                actual = atan2(diff[1], diff[0])
                approx = actual + pi * random.gauss(0, sigma)
                edge.angle = approx
                edge._dest.edges[pt._uid].angle = (pi + approx) % (2*pi)

    def mst(self, edge_weight=(lambda edge: edge.dist)):
        """Make a MST from this network by deleting edges.
        Even edges in the resulting MST will be deleted (and replaced), so
        don't store data on them.
        The edge_weight function should return the weight associated with a
        given edge. Edges with smaller weights will be kept."""

        edges = []
        taken_points = set()
        taken_edges = []

        # pick an arbitrary starting vertex
        taken_points.add(self.points[0])
        for edge in self.points[0].edges.values():
            heapq.heappush(edges, (edge_weight(edge), edge))

        while len(taken_points) < len(self.points):
            __, edge = heapq.heappop(edges)

            if edge._dest in taken_points:
                continue

            taken_edges.append(edge)
            taken_points.add(edge._dest)

            for e in edge._dest.edges.values():
                if e._dest in taken_points:
                    continue
                heapq.heappush(edges, (edge_weight(e), e))

        for pt in self.points:
            pt.edges.clear()

        for edge in taken_edges:
            edge._source.edges[edge._dest._uid] = NetworkEdge(edge._source, edge._dest)
            edge._dest.edges[edge._source._uid] = NetworkEdge(edge._dest, edge._source)

        for pt in self.points:
            pt._order = list(pt.edges)

        self._measure_distances(self._args.sigma)
