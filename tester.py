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


def test_configuration(algorithm_name, point_filename, sigma, args):
    """Test a particular configuration."""

    algo_name, *params = algorithm_name.split(":")

    # Some arguments need to still be included
    if params:
        args.iterations = int(params[0])

    args.sigma = sigma

    algo_func = importlib.import_module(f"algorithms.{algo_name}").solve
    points = read_points_from_file(point_filename)

    total_time = 0
    total_error = 0

    # Count the number of agents in the sample
    num_agents = 0
    for pt in points:
        if pt.typ == "A":
            num_agents += 1

    for runnum in range(args.repeats):
        print(f"Running `{algorithm_name}` on `{point_filename}` with sigma={sigma}. Run {runnum+1}/{args.repeats}")
        start = time.time()
        locations = algo_func(points, args)
        end = time.time()
        total_time += end - start

        for pt, loc in zip(points, locations):
            if pt.typ == "S":
                continue
            err = sum((x-y)*(x-y) for x,y in zip(loc, pt._coords))
            total_error += err

    return [sqrt(total_error / args.repeats / num_agents), total_time / args.repeats]


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
        "-r", "--repeats",
        help="The number of repeats of each run",
        type=int, default=50
    )

    pars.add_argument(
        "-v", "--visibility",
        help="The maximum visible distance",
        required=True,
        type=float
    )

    args = pars.parse_args()

    algorithms = args.algorithms.split(",")
    filenames = args.files.split(",")
    sigmas = list(map(float, args.sigmas.split(",")))

    start = datetime.datetime.now()

    with open("results.csv", "w") as f:

        writer = csv.writer(f)
        writer.writerow(["sample", "algorithm", "sigma", "RMSE", "time"])

        for fname in filenames:
            for algo in algorithms:
                for sigma in sigmas:
                    results = test_configuration(algo, fname, sigma, args)
                    writer.writerow([fname, algo, str(sigma)] + results)

    end = datetime.datetime.now()
    print(f"Simulations concluded. Running time: {str(end-start)}.")
    print("Output written to results.csv.")
