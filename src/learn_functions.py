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


import pickle
import sys
import warnings

import numpy as np
from numpy.random import SFC64

from gp import GP
from util_functions import (
    balance_exploitation,
    early_stop,
    fine_tune_constants,
    fitness,
    live_plot,
    visualize_result,
)

# SEED = None
SEED = 0xFEBA3209B4C18DA4

INSTANCES = {
    0: {
        "POPULATION_SIZE": 400,
        "MAX_DEPTH": 4,
        "MAX_GENERATIONS": 3000,
        "EARLY_STOP_WINDOW_SIZE": 500,
    },
    1: {
        "POPULATION_SIZE": 50,
        "MAX_DEPTH": 2,
        "MAX_GENERATIONS": 50,
        "EARLY_STOP_WINDOW_SIZE": 10,
    },
    2: {
        "POPULATION_SIZE": 300,
        "MAX_DEPTH": 4,
        "MAX_GENERATIONS": 5000,
        "EARLY_STOP_WINDOW_SIZE": 1000,
    },
    3: {
        "POPULATION_SIZE": 300,
        "MAX_DEPTH": 4,
        "MAX_GENERATIONS": 5000,
        "EARLY_STOP_WINDOW_SIZE": 1000,
    },
    4: {
        "POPULATION_SIZE": 300,
        "MAX_DEPTH": 4,
        "MAX_GENERATIONS": 7500,
        "EARLY_STOP_WINDOW_SIZE": 1000,
    },
    5: {
        "POPULATION_SIZE": 300,
        "MAX_DEPTH": 4,
        "MAX_GENERATIONS": 5000,
        "EARLY_STOP_WINDOW_SIZE": 1000,
    },
    6: {
        "POPULATION_SIZE": 300,
        "MAX_DEPTH": 4,
        "MAX_GENERATIONS": 5000,
        "EARLY_STOP_WINDOW_SIZE": 1000,
    },
    7: {
        "POPULATION_SIZE": 300,
        "MAX_DEPTH": 4,
        "MAX_GENERATIONS": 10000,
        "EARLY_STOP_WINDOW_SIZE": 1000,
    },
    8: {
        "POPULATION_SIZE": 400,
        "MAX_DEPTH": 4,
        "MAX_GENERATIONS": 10000,
        "EARLY_STOP_WINDOW_SIZE": 1000,
    },
}


def main(
    filename,
    POPULATION_SIZE,
    MAX_DEPTH,
    MAX_GENERATIONS,
    EARLY_STOP_WINDOW_SIZE,
) -> None:
    problem = np.load(filename)
    # Shuffle the data
    rng = np.random.Generator(SFC64(SEED))
    idx = rng.permutation(problem["x"].shape[-1])

    x = problem["x"][:, idx]
    y = problem["y"][idx]
    # Split the data into training and validation sets
    split = int(0.85 * x.shape[-1])
    x_train = x[:, :split]
    y_train = y[:split]
    x_val = x[:, split:]
    y_val = y[split:]

    gp = GP(x_train, y_train, seed=SEED)

    gp.add_after_loop_hook(lambda _: gp.best.simplify())
    gp.add_after_loop_hook(lambda _: print(f"Best is {gp.best}"))
    gp.add_after_loop_hook(lambda _: print(f"Found in {gp.generation} generations"))
    gp.add_after_loop_hook(
        lambda _: print(f"MSE on training set: {np.mean((gp.best.f(x) - y) ** 2):.3e}")
    )
    gp.add_exploitation_operator("xover", 80)
    # point mutation is quite slower than the other mutation operators

    gp.add_exploration_operator("subtree", 6)
    gp.add_exploration_operator("point", 1)
    gp.add_exploration_operator("hoist", 4)
    gp.add_exploration_operator("permutation", 1)
    gp.add_exploration_operator("collapse", 4)
    gp.add_exploration_operator("expansion", 4)
    gp.set_parent_selector("tournament")
    gp.set_fitness_function(lambda ind: fitness(x_train, y_train, ind))
    gp.set_survivor_selector("fitness_hole")
    gp.add_niching_operator("extinction")
    gp.add_after_iter_hook(lambda gp: balance_exploitation(gp, 100, 0.05))
    gp.add_after_iter_hook(
        lambda gp: fine_tune_constants(
            gp, 0.5, EARLY_STOP_WINDOW_SIZE // 4, 1 + 1e-2, 10
        )
    )
    gp.add_after_iter_hook(lambda gp: early_stop(gp, EARLY_STOP_WINDOW_SIZE, 1 + 1e-5))
    gp.run(
        init_population_size=POPULATION_SIZE,
        init_max_depth=MAX_DEPTH,
        max_generations=MAX_GENERATIONS,
        force_simplify=True,
        parallelize=True,
        use_tqdm=True,
    )

    print(f"MSE on validation set: {np.mean((gp.best.f(x_val) - y_val) ** 2):.3e}")
    with open(f"results/problem_{PROBLEM}.txt", "bw") as f:
        pickle.dump(gp.best, f)

    # Live plot slows everything down and is not recommended unless debugging
    # gp.add_after_iter_hook(lambda gp: live_plot(gp, 5))

    # Draw the best individual as a tree
    # gp.best.draw(block=False)

    # Plot the fitness over generations
    # gp.plot()

    # Plot the best individual projected on the data
    # visualize_result(x, y, gp.best.f, block=False)


def solve(problem):
    filename = f"data/problem_{problem}.npz"
    print(f"Running problem {problem}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main(filename, **INSTANCES[problem])
    print()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        PROBLEM = int(sys.argv[1])
        solve(PROBLEM)
    else:
        for PROBLEM in INSTANCES:
            solve(PROBLEM)
