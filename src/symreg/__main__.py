import argparse
import os
import pwd
import sys

from symreg.learn_functions import main as learn_main
from symreg.visualize_functions import main as visualize_main


def main():
    parser = argparse.ArgumentParser(
        description="Run Genetic Programming solver to learn a function that fits the problems in the `data/` directory by performing symbolic regression. The outputs functions are also saved in the `results/` directory."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Sub-parser for the 'learn' command
    parser_learn = subparsers.add_parser(
        "learn", help="Learn a function that fits the problem(s)"
    )
    parser_learn.add_argument(
        "-p",
        "--profile",
        action="store_true",
        help="Enable profiling",
        required=False,
        default=False,
    )
    parser_learn.add_argument(
        "-m",
        "--multiprocessing",
        action="store_true",
        help="Enable multiprocessing",
        required=False,
        default=False,
    )
    parser_learn.add_argument(
        "-t",
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm",
        default=False,
        required=False,
    )
    parser_learn.add_argument(
        "-l",
        "--live-plot",
        nargs="?",
        type=int,
        help="Enable live plotting (optionally specify the update interval. Default is 50)",
        const=50,
        required=False,
        default=False,
    )
    parser_learn.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite of existing snapshots",
        required=False,
        default=False,
    )
    parser_learn.add_argument(
        "problem", type=int, nargs="?", help="Problem instance to solve", default=None
    )

    # Sub-parser for the 'visualize' command
    parser_visualize = subparsers.add_parser(
        "visualize", help="Visualize the learned function"
    )
    parser_visualize.add_argument(
        "-f",
        "--file",
        type=str,
        default=os.path.join(os.getcwd(), "s327797.py"),
        help="File containing the learned function named `f{problem}` (e.g. `s327797.py`, which contains `f0`, `f1`, etc.) [default: cwd/s327797.py]",
    )
    parser_visualize.add_argument(
        "problem",
        type=int,
        help="Problem instance to visualize",
        default=None,
        # required=True,
    )

    args = parser.parse_args(sys.argv[1:])

    match args.command:
        case None:
            parser.print_help()
            sys.exit(1)
        case "learn":
            learn_main(args)
        case "visualize":
            visualize_main(args)


if __name__ == "__main__":
    main()
