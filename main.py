"""main.py

The main file for simulations.
Run this file with command line args, e.g.
`python main.py -f samples/sample1.csv -a leastsquares -v 5 -s 0.05 -i`

If the `-i` argument is given, draw an image (of a 2D problem) and save it as
image.png.

Algorithms are implemented as .py files in the `algorithms/` subdirectory.
The file should define a function `solve(points, args)`, which solves the
localization problem, defined by the given points.
The `args` object includes all commands line arguments.
Your function should return the estimated positions of all points (including
anchors) in the same order as they were provided.
"""

import argparse
import importlib
import math
import time
from collections import defaultdict

from point import read_points_from_file, Point


def draw_image(points, locations, filename="image.png"):
    """Draw an image of a 2D problem."""
    from PIL import Image, ImageDraw

    # The dimensions (in pixels) of the resulting image.
    WIDTH = 500
    HEIGHT = 500

    PR = 3  # Drawn point radius

    im = Image.new("RGB", (WIDTH, HEIGHT), color=(255,255,255))
    draw = ImageDraw.Draw(im)

    # Find the leftmost, rightmost, highest and lowest coordinates
    # for points. These will be used for coordinate transformation
    left = right = points[0]._coords[0]
    top = bot = points[0]._coords[1]

    for p in points:
        left = min(left, p._coords[0])
        right = max(right, p._coords[0])
        top = max(top, p._coords[1])
        bot = min(bot, p._coords[1])

    def transform(x, y):
        """Transform point coordinates into image coordinates"""
        # Transformation:
        # -WIDTH * 0.45 --- left
        # WIDTH * 0.45 --- right
        # HEIGHT * 0.45 --- top
        # -HEIGHT * 0.45 --- bot
        eta = 0.45
        return (
            round( WIDTH * eta * (2 * x - left - right) / (right - left) + 0.5 * WIDTH ),
            round( HEIGHT * eta * (2 * y - top - bot) / (top - bot) + 0.5 * HEIGHT )
        )

    for p, l in zip(points, locations):
        xp, yp = transform(*p._coords)
        xl, yl = transform(*l)

        if p.typ == "S":
            draw.ellipse([xp-PR, yp-PR, xp+PR, yp+PR], fill=(0,0,255))
        else:
            draw.line([xp, yp, xl, yl], fill=(0,0,0), width=2)
            draw.ellipse([xp-PR, yp-PR, xp+PR, yp+PR], fill=(255, 0, 0))
            draw.ellipse([xl-PR, yl-PR, xl+PR, yl+PR], fill=(0,255,0))

    im.save(filename)


def make_animation(args):
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


def read_and_run(args):
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

    if args.do_image:
        draw_image(points, locations)


def report(args):
    """Report an algorithm's performance and statistics."""
    points = read_points_from_file(args.file)
    solve_function = importlib.import_module(f"algorithms.{args.algorithm}").solve

    runs = args.report
    total_time = 0
    total_error = 0

    num_agents = 0
    for pt in points:
        if pt.typ == "A":
            num_agents += 1

    for runnum in range(runs):
        print(f"\nRun {runnum}:")
        start = time.time()
        locations = solve_function(points, args)
        end = time.time()
        total_time += end - start
        for pt, loc in zip(points, locations):
            if pt.typ == "S":
                continue
            err = sum((x-y)*(x-y) for x,y in zip(loc, pt._coords))
            total_error += err
            print(f"Point {pt} calculated to be at {loc}, error {math.sqrt(err)}.")

    total_error /= runs * num_agents
    print(f"Total RMSE: {math.sqrt(total_error)}")
    print(f"Average time: {total_time / runs}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", help="The file to be read for points", required=True)
    parser.add_argument("-a", "--algorithm", help="The algoritm to use", required=True)
    parser.add_argument("-r", "--report", help="Make a report on the given number of algorithm runs.", type=int, default=0)

    parser.add_argument(
        "-i", "--image",
        help="Draw an image. Currently only supported for 2D coordinates.",
        dest="do_image",
        action="store_const",
        const=True,
        default=False
    )
    parser.add_argument(
        "--animation",
        help="Make an animation. Only supported for 2D coordinates.",
        dest="do_anim",
        action="store_const",
        const=True,
        default=False
    )

    parser.add_argument("-s", "--sigma", help="The standard variation for noise", default=0.05, type=float)
    parser.add_argument("-v", "--visibility", help="The maximum visible distance", type=float, default=math.inf)
    parser.add_argument("-j", "--iterations", help="The number of algorithm iterations, when applicable.", type=int, default=100)

    args = parser.parse_args()

    if args.report > 0:
        report(args)
    elif args.do_anim:
        make_animation(args)
    else:
        read_and_run(args)
