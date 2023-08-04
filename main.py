import argparse
import importlib

from point import read_points_from_file


def draw_image(points, locations):
    from PIL import Image, ImageDraw

    WIDTH = 400
    HEIGHT = 400
    PR = 3  # Drawn point radius

    im = Image.new("RGB", (WIDTH, HEIGHT), color=(255,255,255))
    draw = ImageDraw.Draw(im)

    # Find the leftmost, rightmost, highest and lowest coordinates
    # for points. These will be used for coordinate transformation
    left = right = points[0].coords[0]
    top = bot = points[0].coords[1]

    for p in points:
        left = min(left, p.coords[0])
        right = max(right, p.coords[0])
        top = max(top, p.coords[1])
        bot = min(bot, p.coords[1])


    def transform(x, y):
        """Transform point coordinates into image coordinates"""
        # Transformation:
        # -WIDTH * 0.45 --- left
        # WIDTH * 0.45 --- right
        # HEIGHT * 0.45 --- top
        # -HEIGHT * 0.45 --- bot
        eta = 0.45
        return (
            int( WIDTH * eta * (2 * x - left - right) / (right - left) + 0.5 * WIDTH),
            int( HEIGHT * eta * (2 * y - top - bot) / (top - bot) + 0.5 * HEIGHT)
        )

    for p, l in zip(points, locations):
        xp, yp = transform(*p.coords)
        xl, yl = transform(*l)

        if p.typ == "S":
            draw.ellipse([xp-PR, yp-PR, xp+PR, yp+PR], fill=(0,0,255))
        else:
            draw.line([xp, yp, xl, yl], fill=(0,0,0), width=2)
            draw.ellipse([xp-PR, yp-PR, xp+PR, yp+PR], fill=(255, 0, 0))
            draw.ellipse([xl-PR, yl-PR, xl+PR, yl+PR], fill=(0,255,0))

    im.save("image.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", help="The file to be read for points", required=True)
    parser.add_argument("-a", "--algorithm", help="The algoritm to use", required=True)

    parser.add_argument("-s", "--sigma", help="The standard variation for noise", default=1.0, type=float)
    parser.add_argument(
        "-i", "--image",
        help="Draw an image. Currently only supported for 2D coordinates.",
        dest="do_image",
        action="store_const",
        const=True,
        default=False
    )

    parser.add_argument("-v", "--visibility", help="The maximum visible distance")

    args = parser.parse_args()

    points = read_points_from_file(args.file)
    function = importlib.import_module(f"algorithms.{args.algorithm}").solve

    locations = function(points, args)
    maxerror = 0
    num = 0
    allerror = 0
    for loc, point in zip(locations, points):

        if point.typ == "S":
            continue

        error = point.dist(loc)
        print(f"Point at {point} calculated to be at {loc}, error = {error}")
        maxerror = max(maxerror, error)
        allerror += error
        num += 1

    print(f"Maximal error: {maxerror}")
    print(f"Average error: {allerror / num}")

    if args.do_image:
        draw_image(points, locations)
