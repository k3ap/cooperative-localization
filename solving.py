"""solving.py
Contains code to run and evaluate various algorithms.
"""


import time
import math
from collections import defaultdict
import importlib

from utils import draw_image
from point import read_points_from_file, Point


def read_and_run(args):
    """Evaluate an algorithm's performance on a dataset."""

    # Read and solve the problem
    points = read_points_from_file(args.file)
    solve_function = importlib.import_module(f"algorithms.{args.algorithm}").solve
    locations = solve_function(points, args)

    # Determine the position error.
    maxerror = 0
    num = 0
    errorsum = 0
    for loc, point in zip(locations, points):

        if point.typ == "S":
            continue

        error = point._dist(loc)
        print(f"Point at {point} calculated to be at {loc}, error = {error}")
        maxerror = max(maxerror, error)
        errorsum += error * error
        num += 1

    print(f"Maximal position error: {maxerror}")
    print(f"Position RMSE: {math.sqrt(errorsum / num)}")

    # Determine the distance error
    maxerror = 0
    num = 0
    errorsum = 0
    for loc1, pt1 in zip(locations, points):
        for loc2, pt2 in zip(locations, points):
            error = abs(Point(*loc1)._dist(loc2) - pt1._dist(pt2))
            maxerror = max(maxerror, error)
            errorsum += error * error
            num += 1

    print(f"Maximal distance error: {maxerror}")
    print(f"Distance RMSE: {math.sqrt(errorsum / num)}")

    return points, locations


def make_animation(args):
    """Make an animation for iterative algorithms."""
    import os
    import shutil
    shutil.rmtree("anim/")
    os.mkdir("anim/")
    points = read_points_from_file(args.file)
    solve_function = importlib.import_module(f"algorithms.{args.algorithm}").animate
    for framenum, locations in enumerate(solve_function(points, args)):
        draw_image(points, locations, f"anim/frame{framenum}.png")

    os.chdir("anim/")
    os.system("ffmpeg -framerate 5 -i frame%d.png anim.mp4")
