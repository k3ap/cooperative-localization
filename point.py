from math import sqrt
import csv
import random


class Point:
    def __init__(self, *coords, typ="A"):
        self.typ = typ
        self.coords = coords

    @classmethod
    def from_list(cls, l):
        """Create a new Point from a list containing positions and anchor/agent
        information"""

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
            typ = "A"

        return Point(*coords, typ=typ)

    def abssq(self):
        return sum(x*x for x in self.coords)

    def __abs__(self):
        return sqrt(self.abssq())

    def __iter__(self):
        return iter(self.coords)

    def dist(self, o):
        """Calculate distance to other point or iterable"""
        return sqrt(sum((x-y)*(x-y) for x, y in zip(self, o)))

    def dist_noisy(self, o, sigma):
        return self.dist(o) * (1 + random.gauss(0, sigma))

    def __str__(self):
        return "(" + ", ".join(map(str, self.coords)) + ")"


def read_points_from_file(filename: str):
    points = []
    with open(filename) as f:
        reader = csv.reader(f)
        for ln in reader:
            points.append(Point.from_list(ln))

    return points
