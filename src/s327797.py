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


import warnings

import numpy as np

from gp import GP
import sys


sys.setrecursionlimit(2000)
SEED = 0
PROBLEM = 2

POPULATION_SIZE = 40
MAX_DEPTH = 4
MAX_GENERATIONS = 1000


def fitness(x, y, ind, weights: tuple):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mse = np.mean((ind.f(x) - y) ** 2)
    fitness = weights[0] / mse - weights[1] * ind.depth

    if np.isnan(fitness) or np.isinf(fitness):
        return 0

    return fitness


if __name__ == "__main__":
    problem = np.load(f"tests/problem_{PROBLEM}.npz")
    x = problem["x"]
    y = problem["y"]
    gp = GP(x, y, seed=SEED, use_tqdm=True)

    gp.add_before_loop_hook(lambda: print(f"Starting on problem {PROBLEM}"))
    gp.add_after_loop_hook(lambda: print(f"Finished on problem {PROBLEM}"))
    gp.add_after_loop_hook(lambda: print(f"Best is {gp.best.simplify(x)}"))
    gp.add_after_loop_hook(lambda: print(f"Found in {gp.generation} generations"))
    gp.add_after_loop_hook(
        lambda: print(
            f"MSE on training set: {np.mean((gp.best.simplify(x).f(x) - y) ** 2):.3e}"
        )
    )
    gp.add_genetic_operator("xover", 0.9)
    gp.add_genetic_operator("point", 0.01)
    gp.add_genetic_operator("permutation", 0.09)
    gp.set_parent_selector("fitness_proportional")
    gp.set_fitness_function(lambda ind: fitness(x, y, ind, (0.95, 0.05)))
    gp.set_survivor_selector("deterministic")
    gp.add_niching_operator("extinction")

    gp.run(
        init_population_size=POPULATION_SIZE,
        init_max_depth=MAX_DEPTH,
        max_generations=MAX_GENERATIONS,
    )

    validation = np.load(f"tests/validation_{PROBLEM}.npz")
    x_val = validation["x"]
    y_val = validation["y"]
    print(
        f"MSE on validation set: {np.mean((gp.best.simplify(x).f(x_val) - y_val) ** 2):.3e}"
    )

    try:
        gp.best.simplify(x).draw()
    except KeyboardInterrupt:
        pass
