"""visualize.py
Visualize the network graph of a sample, with given visibility.

Example usage:
`python visualize.py -f samples/standard/sample1.csv -v 0.5`
"""


import argparse
from PIL import Image, ImageDraw

from network import Network, NetworkNode
from point import read_points_from_file


def draw_sample_image(sample_filename, args, image_filename="image.png"):

    network = Network(read_points_from_file(sample_filename), NetworkNode, args, check_disconnect=False)
    if args.mst:
        network.mst()

    # The dimensions (in pixels) of the resulting image.
    WIDTH = 500
    HEIGHT = 500

    PR = 3  # Drawn point radius

    im = Image.new("RGB", (WIDTH, HEIGHT), color=(255,255,255))
    draw = ImageDraw.Draw(im)

    # Find the leftmost, rightmost, highest and lowest coordinates
    # for points. These will be used for coordinate transformation
    left = right = network.points[0]._coords[0]
    top = bot = network.points[0]._coords[1]

    for p in network.points:
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

    for p in network.points:
        xp, yp = transform(*p._coords)
        for edge in p.edges.values():
            xe, ye = transform(*edge._dest._coords)
            draw.line([xp, yp, xe, ye], fill=(0,0,0), width=2)

    for p in network.points:
        xp, yp = transform(*p._coords)

        if p.typ == "S":
            draw.ellipse([xp-PR, yp-PR, xp+PR, yp+PR], fill=(0,0,255))
        else:
            draw.ellipse([xp-PR, yp-PR, xp+PR, yp+PR], fill=(255, 0, 0))

    im.save(image_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", help="The sample file to visualize.")
    parser.add_argument("-v", "--visibility", help="The visibility distance.", type=float)
    parser.add_argument("-mst", help="Draw the minimum spanning tree.", default=False, action="store_const", const=True)

    args = parser.parse_args()
    args.sigma = 0

    draw_sample_image(args.file, args)
