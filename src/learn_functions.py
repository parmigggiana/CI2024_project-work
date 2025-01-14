try:
    from icecream import install

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    try:
        builtins = __import__("__builtin__")
    except ImportError:
        builtins = __import__("builtins")
    setattr(
        builtins,
        "ic",
        lambda *a: (None if not a else (a[0] if len(a) == 1 else a)),
    )


import argparse
import datetime
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from numpy.random import SFC64

from gp import GP
from util_functions import (
    balance_exploitation,
    early_stop,
    fine_tune_constants,
    fitness,
    visualize_result,
)
from util_functions import live_plot as live_plot_fn

# SEED = None
SEED = 0xFEBA3209B4C18DA4

INSTANCES = {
    0: {
        "POPULATION_SIZE": 400,
        "MAX_DEPTH": 4,
        "MAX_GENERATIONS": 1600,
        "EARLY_STOP_WINDOW_SIZE": 400,
    },
    1: {
        "POPULATION_SIZE": 100,
        "MAX_DEPTH": 3,
        "MAX_GENERATIONS": 100,
        "EARLY_STOP_WINDOW_SIZE": 25,
    },
    2: {
        "POPULATION_SIZE": 500,
        "MAX_DEPTH": 5,
        "MAX_GENERATIONS": 5000,
        "EARLY_STOP_WINDOW_SIZE": 2000,
    },
    3: {
        "POPULATION_SIZE": 500,
        "MAX_DEPTH": 5,
        "MAX_GENERATIONS": 500,
        "EARLY_STOP_WINDOW_SIZE": 100,
    },
    4: {
        "POPULATION_SIZE": 500,
        "MAX_DEPTH": 6,
        "MAX_GENERATIONS": 2500,
        "EARLY_STOP_WINDOW_SIZE": 1000,
    },
    5: {
        "POPULATION_SIZE": 400,
        "MAX_DEPTH": 4,
        "MAX_GENERATIONS": 2500,
        "EARLY_STOP_WINDOW_SIZE": 1000,
    },
    6: {
        "POPULATION_SIZE": 400,
        "MAX_DEPTH": 4,
        "MAX_GENERATIONS": 500,
        "EARLY_STOP_WINDOW_SIZE": 10,
    },
    7: {
        "POPULATION_SIZE": 400,
        "MAX_DEPTH": 4,
        "MAX_GENERATIONS": 4000,
        "EARLY_STOP_WINDOW_SIZE": 1000,
    },
    8: {
        "POPULATION_SIZE": 500,
        "MAX_DEPTH": 5,
        "MAX_GENERATIONS": 8000,
        "EARLY_STOP_WINDOW_SIZE": 2000,
    },
}


def save_snapshot(gp, filename, function_only=False):
    os.makedirs("results", exist_ok=True)
    with open(filename, "bw") as fs:
        if function_only:
            pickle.dump(gp.best.f, fs)
        else:
            pickle.dump(gp.population, fs)


def restore_snapshot(gp, filename):
    if Path(filename).exists():
        with open(filename, "br") as fs:
            gp.population = pickle.load(fs)


def solve(
    problem,
    multiprocessing=False,
    tqdm=True,
    live_plot: bool | int = False,
    ignore_snapshots=False,
    profile=False,
) -> None:
    filename = f"data/problem_{problem}.npz"
    print(f"Running problem {problem}")

    instance = INSTANCES[problem]
    POPULATION_SIZE = instance["POPULATION_SIZE"]
    MAX_DEPTH = instance["MAX_DEPTH"]
    MAX_GENERATIONS = instance["MAX_GENERATIONS"]
    EARLY_STOP_WINDOW_SIZE = instance["EARLY_STOP_WINDOW_SIZE"]

    problem_data = np.load(filename)
    # Shuffle the data
    rng = np.random.Generator(SFC64(SEED))
    idx = rng.permutation(problem_data["x"].shape[-1])

    x = problem_data["x"][:, idx]
    y = problem_data["y"][idx]
    # Split the data into training and validation sets
    split = int(0.85 * x.shape[-1])
    x_train = x[:, :split]
    y_train = y[:split]
    x_val = x[:, split:]
    y_val = y[split:]

    gp = GP(x_train, y_train, seed=SEED)

    gp.add_exploitation_operator("xover", 80)
    gp.add_exploration_operator("subtree", 4)
    gp.add_exploration_operator("point", 1)
    gp.add_exploration_operator("hoist", 1)
    gp.add_exploration_operator("permutation", 2)
    gp.add_exploration_operator("collapse", 7)
    gp.add_exploration_operator("expansion", 5)
    gp.set_fitness_function(lambda ind: fitness(x_train, y_train, ind))
    gp.set_parent_selector("tournament")
    gp.set_survivor_selector("fitness_hole")
    gp.add_niching_operator("extinction")
    if not ignore_snapshots:
        gp.add_before_loop_hook(
            lambda gp: restore_snapshot(gp, f"snapshot_{problem}.tmp")
        )
    gp.add_after_iter_hook(lambda gp: balance_exploitation(gp, 50, 0.05))
    gp.add_after_iter_hook(lambda gp: early_stop(gp, EARLY_STOP_WINDOW_SIZE, 1.2))
    gp.add_after_iter_hook(
        lambda gp: (gp.generation % 50 == 0)
        and save_snapshot(gp, f"snapshot_{problem}.tmp")
    )
    gp.add_after_loop_hook(lambda gp: gp.best.simplify())
    gp.add_after_loop_hook(fine_tune_constants)
    gp.add_after_loop_hook(lambda gp: print(f"Best is {gp.best}"))
    gp.add_after_loop_hook(lambda gp: print(f"Found in {gp.generation} generations"))
    gp.add_after_loop_hook(
        lambda _: print(f"MSE on training set: {np.mean((gp.best.f(x) - y) ** 2):.3e}")
    )

    # Not recommended unless debugging
    if live_plot:
        gp.add_after_iter_hook(
            lambda gp: live_plot_fn(gp, live_plot, max_individuals=16)
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if profile:
            import cProfile
            import pstats

            pr = cProfile.Profile()
            pr.enable()
        gp.run(
            init_population_size=POPULATION_SIZE,
            init_max_depth=MAX_DEPTH,
            max_generations=MAX_GENERATIONS,
            force_simplify=True,
            parallelize=multiprocessing,
            use_tqdm=tqdm,
        )
        if profile:
            pr.disable()
            stats = pstats.Stats(pr)
            stats.strip_dirs()
            stats.sort_stats(pstats.SortKey.TIME)
            # stats.sort_stats(pstats.SortKey.CUMULATIVE)
            stats.dump_stats(f"run_{datetime.datetime.now().isoformat()}.prof")

    save_snapshot(gp, f"results/problem_{problem}", function_only=True)

    print(f"MSE on validation set: {np.mean((gp.best.f(x_val) - y_val) ** 2):.3e}")
    print()

    # Plot the fitness over generations
    gp.plot()

    # Draw the best individual as a tree
    gp.best.draw(block=False)

    # Plot the best individual projected on the data
    visualize_result(x, y, gp.best.f, block=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run Genetic Programming solver to learn a function that fits the problems in the `data/` directory by performing symbolic regression. The outputs functions are also saved in the `results/` directory."
    )
    parser.add_argument(
        "-p",
        "--profile",
        action="store_true",
        help="Enable profiling",
        required=False,
        default=False,
    )
    parser.add_argument(
        "-m",
        "--multiprocessing",
        action="store_true",
        help="Enable multiprocessing",
        required=False,
        default=False,
    )
    parser.add_argument(
        "-t",
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-l",
        "--live-plot",
        nargs="?",
        type=int,
        help="Enable live plotting (optionally specify the update interval. Default is 50)",
        const=50,
        required=False,
        default=False,
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite of existing snapshots",
        required=False,
        default=False,
    )
    parser.add_argument(
        "problem", type=int, nargs="?", help="Problem instance to solve", default=None
    )

    args = parser.parse_args(sys.argv[1:])

    for instance in INSTANCES:
        if args.problem != instance:
            continue
        solve(
            problem=instance,
            multiprocessing=args.multiprocessing,
            tqdm=not args.no_tqdm,
            live_plot=args.live_plot,
            ignore_snapshots=args.force,
            profile=args.profile,
        )


if __name__ == "__main__":
    main()
