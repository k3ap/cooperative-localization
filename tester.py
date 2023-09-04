"""tester.py
Compare and evaluate different algorithms on different samples.
This is a standalone file designed primarily to compare algorithms.

Example usage:
`python tester.py -a admmh:50,convexrelaxation:200,leastsquarescoop -f samples/standard/sample1.csv,samples/standard/sample2.csv -v 0.5 -r 30 -s 0,0.1,0.25`
"""


import argparse
import csv
import time
from math import sqrt
import importlib
import datetime

from point import read_points_from_file
from network import DisconnectedGraphError
from samples.standard.generate_random import generate_points


def test_configuration(algorithm_name, point_filename, sigma, sigma_angles, visibility, args):
    """Test a particular configuration."""

    algo_name, *params = algorithm_name.split(":")

    # Some arguments need to still be included
    if params:
        args.iterations = int(params[0])

    args.sigma = sigma
    args.sigma_angles = sigma_angles
    args.visibility = visibility

    algo_func = importlib.import_module(f"algorithms.{algo_name}").solve

    total_time = 0
    total_error = 0

    runs = 0

    for runnum in range(args.repeats):
        print(f"Running `{algorithm_name}` on `{point_filename}` with sigmas={sigma},{sigma_angles} and v={visibility}. Run {runnum+1}/{args.repeats}")

        run_error = 0

        if point_filename.startswith("RANDOM"):
            __, num_anchors, num_agents = point_filename.split(":")
            num_anchors = int(num_anchors)
            num_agents = int(num_agents)
            with open("RANDOM.csv", "w") as f:
                writer = csv.writer(f)
                lines = generate_points(num_anchors, num_agents, 0.05)
                writer.writerows(lines)

        points = read_points_from_file(
            point_filename if not point_filename.startswith("RANDOM") else "RANDOM.csv"
        )

        # Count the number of agents in the sample
        num_agents = 0
        for pt in points:
            if pt.typ == "A":
                num_agents += 1

        start = time.time()
        try:
            locations = algo_func(points, args)
        except DisconnectedGraphError:
            print("Disconnected graph! Is visibility set too low? Skipping this run.")
            continue

        end = time.time()
        total_time += end - start
        runs += 1

        for pt, loc in zip(points, locations):
            if pt.typ == "S":
                continue
            err = sum((x-y)*(x-y) for x,y in zip(loc, pt._coords))
            run_error += err

        total_error += run_error / num_agents

    return [sqrt(total_error / runs), total_time / args.repeats]


if __name__ == "__main__":
    pars = argparse.ArgumentParser()

    pars.add_argument(
        "-a", "--algorithms",
        help="A comma-separated list of algorithms to compare. In case of iterative algorithms, write a colon, followed by the number of iterations",
        required=True
    )
    pars.add_argument(
        "-f", "--files",
        help="A comma-separated list of samples to compare",
        required=True
    )
    pars.add_argument(
        "-s", "--sigmas",
        help="A comma-separated list of sigma values to compare",
        required=True
    )
    pars.add_argument(
        "-t", "--sigmas-angles",
        help="A comma-separated list of angle sigma values to compare",
        default="0.03"
    )

    pars.add_argument(
        "-r", "--repeats",
        help="The number of repeats of each run",
        type=int, default=50
    )

    pars.add_argument(
        "-v", "--visibilities",
        help="The maximum visible distances to test.",
        required=True
    )

    args = pars.parse_args()

    algorithms = args.algorithms.split(",")
    filenames = args.files.split(",")
    sigmas = list(map(float, args.sigmas.split(",")))
    sigmas_angles = list(map(float, args.sigmas_angles.split(",")))
    visibilites = list(map(float, args.visibilities.split(",")))

    start = datetime.datetime.now()

    with open("results.csv", "w") as f:

        writer = csv.writer(f)
        writer.writerow(["sample", "algorithm", "sigma", "sigma_angle", "visibility", "RMSE", "time"])

        for fname in filenames:
            for algo in algorithms:
                for sigma in sigmas:
                    for sigma_angle in sigmas_angles:
                        for visibility in visibilites:
                            results = test_configuration(algo, fname, sigma, sigma_angle, visibility, args)
                            writer.writerow([fname, algo, str(sigma), str(sigma_angle), str(visibility)] + results)

    end = datetime.datetime.now()
    print(f"Simulations concluded. Running time: {str(end-start)}.")
    print("Output written to results.csv.")
