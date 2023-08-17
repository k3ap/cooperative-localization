"""distrib.py
Contains the definition of the NetworkPoint class, the base class for old
distributed simulations.
To use a distributed simulation, you should subclass NetworkPoint, and convert
the given points into your subclass in the `solve` function of your algorithm.
Afterwards, call `add_neighbours` on all points, followed by `measure_distances`
on all points.
Implement the actual algorithm in your subclass, and call those methods in
the desired order in the `solve` function.
See `algorithms/leastsquaresdistrib.py` for an example usage.

This class is deprecated. Use network.py instead of it.
"""


from collections import deque

from point import Point


class NetworkPoint(Point):
    def __init__(self, point):
        super().__init__(*point._coords, typ=point.typ)
        self.neighbours = []
        self.distances = []
        self.message_queue = deque()

    def add_neighbours(self, points, visibility):
        """Add the visible points from the given list to neighbours"""
        for point in points:
            if point._dist(self) < visibility and point is not self:
                self.neighbours.append(point)
                self.distances.append(None)

    def measure_distances(self, sigma):
        """Measure (and synchronize) distances to neighbours."""
        for i, pt in enumerate(self.neighbours):
            if self.distances[i] is None:
                self.distances[i] = self.dist_noisy(pt, sigma)
                pt.set_distance(self, self.distances[i])

    def set_distance(self, to, distance):
        """Set the distance to the given neighbour"""
        for i, pt in enumerate(self.neighbours):
            if pt is to:
                self.distances[i] = distance

    def __str__(self):
        return "NP(" + ", ".join(map(str, self.coords)) + ")"

    def receive(self, message, sender):
        """Receive a message. You should handle messages in another function,
        which reads from the message queue."""
        self.message_queue.append((message, sender))

    def broadcast(self, message):
        """Broadcast a message to all neighbouring nodes"""
        for pt in self.neighbours:
            pt.receive(message, self)

    def send(self, message, to):
        """Send a message"""
        if to not in self.neighbours:
            return
        to.receive(message, self)
