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
    build_fitness_func,
    early_stop,
    fine_tune_constants,
)
from util_functions import live_plot as live_plot_fn
from util_functions import visualize_result

SEED = None
# SEED = 0xFEBA3209B4C18DA4

INSTANCES = {
    0: {
        "POPULATION_SIZE": 400,
        "MAX_DEPTH": 4,
        "MAX_GENERATIONS": 1600,
        "EARLY_STOP_WINDOW_SIZE": 400,
        "FITNESS_FUNCTION": lambda ind, mse: 1 / mse / np.sqrt(len(ind.nodes)),
    },
    1: {
        "POPULATION_SIZE": 100,
        "MAX_DEPTH": 3,
        "MAX_GENERATIONS": 100,
        "EARLY_STOP_WINDOW_SIZE": 25,
        "FITNESS_FUNCTION": lambda ind, mse: 1 / mse / np.sqrt(len(ind.nodes)),
    },
    2: {
        "POPULATION_SIZE": 700,
        "MAX_DEPTH": 5,
        "MAX_GENERATIONS": 300,
        "EARLY_STOP_WINDOW_SIZE": 100,
        "FITNESS_FUNCTION": lambda ind, mse: 1 / mse - 0.001 * len(ind.nodes),
    },
    3: {
        "POPULATION_SIZE": 500,
        "MAX_DEPTH": 5,
        "MAX_GENERATIONS": 2000,
        "EARLY_STOP_WINDOW_SIZE": 400,
        "FITNESS_FUNCTION": lambda ind, mse: 1 / mse / np.sqrt(len(ind.nodes)),
    },
    4: {
        "POPULATION_SIZE": 500,
        "MAX_DEPTH": 5,
        "MAX_GENERATIONS": 1500,
        "EARLY_STOP_WINDOW_SIZE": 400,
        "FITNESS_FUNCTION": lambda ind, mse: 1 / mse / np.sqrt(len(ind.nodes)),
    },
    5: {
        "POPULATION_SIZE": 1000,
        "MAX_DEPTH": 5,
        "MAX_GENERATIONS": 3000,
        "EARLY_STOP_WINDOW_SIZE": 3000,
        "FITNESS_FUNCTION": lambda ind, mse: 1 / mse,  # - 1e-9 * len(ind.nodes),
    },
    6: {
        "POPULATION_SIZE": 500,
        "MAX_DEPTH": 5,
        "MAX_GENERATIONS": 500,
        "EARLY_STOP_WINDOW_SIZE": 300,
        "FITNESS_FUNCTION": lambda ind, mse: 1 / mse - 0.06 * len(ind.nodes),
    },
    7: {
        "POPULATION_SIZE": 500,
        "MAX_DEPTH": 5,
        "MAX_GENERATIONS": 1000,
        "EARLY_STOP_WINDOW_SIZE": 50,
        "FITNESS_FUNCTION": lambda ind, mse: 1 / mse - 0.0001 * len(ind.nodes),
    },
    8: {
        "POPULATION_SIZE": 500,
        "MAX_DEPTH": 5,
        "MAX_GENERATIONS": 4000,
        "EARLY_STOP_WINDOW_SIZE": 2,
        "FITNESS_FUNCTION": lambda ind, mse: 1 / mse - 0.0001 * len(ind.nodes),
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
            snapshot = pickle.load(fs)
            snapshot_length = len(snapshot)
            if snapshot_length >= gp.population_size:
                gp.population = snapshot[: gp.population_size]
            else:
                gp.population[:snapshot_length] = snapshot


def balance_weights(gp, mod=1, takeover_len=50):
    if gp.generation % mod != 0:
        return

    sample_len = np.percentile([len(ind.nodes) for ind in gp.population], 90)

    base_w = [1, 6, 1, 4, 1, 2, 5]
    half_point_w = [4, 8, 1, 3, 1, 2, 1]
    takeover_w = [3, 15, 1, 1, 0, 0, 0]

    if sample_len < takeover_len // 2:
        new_weights = [
            w + (h_w - w) * (sample_len - 1) / (takeover_len // 2 - 1)
            for w, h_w in zip(base_w, half_point_w)
        ]
    elif sample_len < takeover_len:
        new_weights = [
            w
            + (t_w - w)
            * (sample_len - takeover_len // 2)
            / (takeover_len - takeover_len // 2)
            for w, t_w in zip(half_point_w, takeover_w)
        ]
    else:
        new_weights = takeover_w

    # convert to integers but keep the sum the same
    # int_weights = [int(w) for w in new_weights]
    # diff = sum(base) - sum(int_weights)
    # while diff > 0:
    #     max_idx = np.argmax([w - i for i, w in zip(int_weights, new_weights)])
    #     int_weights[max_idx] += 1
    #     new_weights[max_idx] = int_weights[max_idx]
    #     diff -= 1

    # print(f"New weights: {int_weights} - Sample depth: {sample_depth:.2f}")
    gp._exploration_operators_weights = new_weights


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
    split = int(0.75 * x.shape[-1])
    x_train = x[:, :split]
    y_train = y[:split]
    x_val = x[:, split:]
    y_val = y[split:]

    gp = GP(x_train, y_train, seed=SEED)

    mutations = {
        "hoist": 1,
        "shrink": 6,
        "collapse": 1,
        "subtree": 4,
        "point": 1,
        "permutation": 2,
        "expansion": 5,
    }

    gp.add_exploitation_operator("xover", 80)
    for k, v in mutations.items():
        gp.add_exploration_operator(k, v)
    gp.set_fitness_function(
        build_fitness_func(x_train, y_train, instance["FITNESS_FUNCTION"])
    )
    gp.set_parent_selector("tournament")
    gp.set_survivor_selector("fitness_hole")
    # gp.add_niching_operator("extinction")
    if not ignore_snapshots:
        gp.add_before_loop_hook(
            lambda gp: restore_snapshot(gp, f"snapshot_{problem}.tmp")
        )
    gp.add_before_iter_hook(balance_weights)
    # gp.add_after_iter_hook(lambda gp: balance_exploitation(gp, 50, 0.05))
    gp.add_after_iter_hook(lambda gp: early_stop(gp, EARLY_STOP_WINDOW_SIZE, 1.002))
    gp.add_after_iter_hook(
        lambda gp: (gp.generation % 50 == 0)
        and save_snapshot(gp, f"snapshot_{problem}.tmp")
    )
    gp.add_after_loop_hook(lambda gp: gp.best.simplify())
    gp.add_after_loop_hook(fine_tune_constants)
    gp.add_after_loop_hook(
        lambda gp: print(
            f"Best has depth {gp.best.depth} ({len(gp.best.nodes)} nodes):\n{gp.best}"
        )
    )
    gp.add_after_loop_hook(lambda gp: print(f"Found in {gp.generation} generations"))
    gp.add_after_loop_hook(
        lambda _: print(f"MSE on training set: {np.mean((gp.best.f(x) - y) ** 2):.3e}")
    )

    # Not recommended unless debugging
    if live_plot:
        gp.add_after_iter_hook(
            lambda gp: live_plot_fn(gp, live_plot, max_individuals=12)
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
    gp.plot(block=False)

    # Draw the best individual as a tree
    gp.best.draw(block=False)

    # Plot the best individual projected on the data
    visualize_result(x, y, gp.best.f, block=True)


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
