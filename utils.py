"""utils.py
Various utilities."""


import numpy as np
import random
import string


def collect_anchors_and_distances(point, points, visibility, sigma, min_anchors=3):
    """Find anchors in range and generate randomized distances to them"""
    anchors = [p for p in points if p.typ == "S" and p._dist(point) < visibility]
    if len(anchors) < min_anchors:
        raise ValueError("Not enough anchors")
    distances = [a.dist_noisy(point, sigma) for a in anchors]
    return anchors, distances


def random_vector(spans):
    """Return a random vector close to the problem bounds."""
    return np.matrix([
        random.uniform(mini-1, maxi+1) for mini, maxi in spans
    ]).T


GENERATED_UIDS = set()

def generate_uid(length=16):
    """Generate a unique identifier of specified length."""
    def gen_string(l):
        s = ""
        for __ in range(l):
            s += random.choice(string.ascii_letters)
        return s

    while (s := gen_string(length)) in GENERATED_UIDS:
        pass

    GENERATED_UIDS.add(s)
    return s


def draw_image_smart(points, locations):
    """Draw either an image or a 3d plot, depending on point dimensions."""
    if points[0].dim == 2:
        draw_image(points, locations)
    elif points[0].dim == 3:
        draw_3d_plot(points, locations)
    else:
        raise ValueError(f"Cannot draw {points[0].dim}-dimensional image.")


def draw_3d_plot(points, locations):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    xs_anchors = [pt._coords[0] for pt in points if pt.typ == "S"]
    ys_anchors = [pt._coords[1] for pt in points if pt.typ == "S"]
    zs_anchors = [pt._coords[2] for pt in points if pt.typ == "S"]

    xs_agents = [pt._coords[0] for pt in points if pt.typ == "A"]
    ys_agents = [pt._coords[1] for pt in points if pt.typ == "A"]
    zs_agents = [pt._coords[2] for pt in points if pt.typ == "A"]

    xs_locs = [loc[0] for pt, loc in zip(points, locations) if pt.typ == "A"]
    ys_locs = [loc[1] for pt, loc in zip(points, locations) if pt.typ == "A"]
    zs_locs = [loc[2] for pt, loc in zip(points, locations) if pt.typ == "A"]

    for pt, loc in zip(points, locations):
        if pt.typ == "S":
            continue
        ax.plot([pt._coords[0], loc[0]], [pt._coords[1], loc[1]], [pt._coords[2], loc[2]], c="black")

    ax.scatter(xs_anchors, ys_anchors, zs_anchors, c="blue")
    ax.scatter(xs_agents, ys_agents, zs_agents, c="red")
    ax.scatter(xs_locs, ys_locs, zs_locs, c="green")

    plt.show()



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
