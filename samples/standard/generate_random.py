"""generate_random.py
Generate a random standardized sample.
Standardized samples have a number of anchors evenly spread out at the edge
of a unit square, and a number of agents randomly spread within.
"""


import random
import argparse
import csv


def edge_parametrization(t):
    """Parametrization of the edge of the unit square."""
    if t > 3:
        return (0, 4 - t)
    if t > 2:
        return (3 - t, 1)
    if t > 1:
        return (1, t - 1)
    return (t, 0)


def generate_points(num_anchors, num_agents, margin):
    lines = []

    # Make the anchors
    for i in range(num_anchors):
        lines.append((*edge_parametrization(4 * i / num_anchors), "S"))

    # Make the agents
    for __ in range(num_agents):
        lines.append((
            random.uniform(margin, 1-margin),
            random.uniform(margin, 1-margin),
            "A"
        ))

    return lines


if __name__ == "__main__":
    pars = argparse.ArgumentParser()
    pars.add_argument("-s", help="Set the number of anchors", dest="anchors", type=int, default=4)
    pars.add_argument("-a", help="Set the number of agents", dest="agents", type=int, default=30)
    pars.add_argument("-m", help="The margin around the edge, where no agents are made", type=float, default=0.05, dest="margin")
    pars.add_argument("-f", help="The file to save to", dest="filename", required=True)

    args = pars.parse_args()

    lines = generate_points(args.anchors, args.agents, args.margin)

    with open(args.filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(lines)
