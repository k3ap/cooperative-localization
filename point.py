"""point.py
The definition of the Point class, the basic object of the simulations."""


from math import sqrt
import csv
import random


class Point:
    def __init__(self, *coords, typ="A"):
        """Initialize a point. You must provide coordinates (the dimension will
        be determined from the number of coordinates provided) and a type.
        Type 'S' means that the point is an anchor (sidro),
        while type 'A' means that the point is an agent."""
        self.typ = typ
        self.coords = coords

    @classmethod
    def from_list(cls, l):
        """Create a new Point from a list containing positions and anchor/agent
        information. Used while reading a Point from a file."""

        coords = []
        typ = None

        for p in l:
            try:
                num = float(p)
                coords.append(num)
            except ValueError:
                typ = p
                break
        else:
            # if there is no type, assume the point to be an agent
            typ = "A"

        return Point(*coords, typ=typ)

    def abssq(self):
        """The square of the absolute value."""
        return sum(x*x for x in self.coords)

    def __abs__(self):
        return sqrt(self.abssq())

    def __iter__(self):
        return iter(self.coords)

    def dist(self, o):
        """Calculate distance to other point or iterable"""
        return sqrt(sum((x-y)*(x-y) for x, y in zip(self, o)))

    def dist_noisy(self, o, sigma):
        """Calculate a noisy distance to other point or iterable."""
        return self.dist(o) * (1 + random.gauss(0, sigma))

    def __str__(self):
        return "P(" + ", ".join(map(str, self.coords)) + ")"


def read_points_from_file(filename: str):
    """Read points from a CSV file. See samples/*.csv for file structure."""
    points = []
    with open(filename) as f:
        reader = csv.reader(f)
        for ln in reader:
            points.append(Point.from_list(ln))

    return points
