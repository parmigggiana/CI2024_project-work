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


import sys

import numpy as np
from numpy.random import SFC64

from gp import GP
from util_functions import (
    early_stop,
    fine_tune_constants,
    fitness,
    live_plot,
    visualize_data,
    visualize_result,
)

# sys.setrecursionlimit(5000)
SEED = 0xFEBA3209B4C18DA4
PROBLEM = 0

POPULATION_SIZE = 200
MAX_DEPTH = 5

MAX_GENERATIONS = 2000
EARLY_STOP_WINDOW_SIZE = 300


filename = f"data/problem_{PROBLEM}.npz"

if __name__ == "__main__":
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

    # fig, ax = visualize_data(x, y, block=False)

    gp = GP(x_train, y_train, seed=SEED)

    gp.add_before_loop_hook(lambda _: print(f"Starting on problem {PROBLEM}"))
    gp.add_after_loop_hook(lambda _: print(f"Finished on problem {PROBLEM}"))
    gp.add_after_loop_hook(lambda _: gp.best.simplify())
    gp.add_after_loop_hook(lambda _: print(f"Best is {gp.best}"))
    gp.add_after_loop_hook(lambda _: print(f"Found in {gp.generation} generations"))
    gp.add_after_loop_hook(
        lambda _: print(f"MSE on training set: {np.mean((gp.best.f(x) - y) ** 2):.3e}")
    )
    gp.add_exploitation_operator("xover", 40)
    # point mutation is quite slower than the other mutation operators
    gp.add_exploration_operator("point", 1)
    gp.add_exploration_operator("hoist", 2)
    gp.add_exploration_operator("permutation", 5)
    gp.set_parent_selector("fitness_proportional")
    gp.set_fitness_function(lambda ind: fitness(x_train, y_train, ind))
    gp.set_survivor_selector("deterministic")
    gp.add_niching_operator("extinction")
    gp.add_after_iter_hook(lambda gp: gp.change_exploitation_bias(50, 0.2))
    gp.add_after_iter_hook(
        lambda gp: fine_tune_constants(
            gp, 0.90, EARLY_STOP_WINDOW_SIZE // 3, 1 + 1e-5, 25
        )
    )
    gp.add_after_iter_hook(lambda gp: early_stop(gp, EARLY_STOP_WINDOW_SIZE, 1 + 1e-5))
    # Live plot slows everything down and is not recommended for large population sizes
    # gp.add_after_iter_hook(lambda gp: live_plot(gp, 50))
    gp.run(
        init_population_size=POPULATION_SIZE,
        init_max_depth=MAX_DEPTH,
        max_generations=MAX_GENERATIONS,
        force_simplify=True,
        parallelize=True,
        use_tqdm=True,
    )

    visualize_result(x, y, gp.best.f, block=False)

    print(f"MSE on validation set: {np.mean((gp.best.f(x_val) - y_val) ** 2):.3e}")

    try:
        gp.best.draw(block=False)
        gp.plot()
    except KeyboardInterrupt:
        pass
