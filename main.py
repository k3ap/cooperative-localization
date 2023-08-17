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
import math

from utils import draw_image
from solving import make_animation, read_and_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", help="The file to be read for points", required=True)
    parser.add_argument("-a", "--algorithm", help="The algoritm to use", required=True)

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

    if args.do_anim:
        make_animation(args)
    else:
        pts, locs = read_and_run(args)
        if args.do_image:
            draw_image(pts, locs)

