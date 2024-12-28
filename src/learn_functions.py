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

from gp import GP
from util_functions import early_stop, fitness, live_plot, visualize_data, visualize_result

sys.setrecursionlimit(5000)
SEED = 42
PROBLEM = 0

POPULATION_SIZE = 500
MAX_DEPTH = 5
MAX_GENERATIONS = 2000

filename = f"data/problem_{PROBLEM}.npz"

if __name__ == "__main__":
    problem = np.load(filename)
    # Shuffle the data
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(problem["x"].shape[-1])

    x = problem["x"][:,idx]
    y = problem["y"][idx]
    # Split the data into training and validation sets
    split = int(0.85 * x.shape[-1])
    x_train = x[:,:split]
    y_train = y[:split]
    x_val = x[:,split:]
    y_val =y[split:]

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
    gp.add_genetic_operator("xover", 0.8)
    gp.add_genetic_operator("point", 0.06)
    gp.add_genetic_operator("hoist", 0.06)
    gp.add_genetic_operator("permutation", 0.08)
    gp.set_parent_selector("fitness_proportional")
    gp.set_fitness_function(lambda ind: fitness(x_train, y_train, ind, (0.9, 0.1)))
    gp.set_survivor_selector("deterministic")
    gp.add_niching_operator("extinction")
    gp.add_after_iter_hook(lambda gp: early_stop(gp, 600, 1 + 1e-5))
    # gp.add_after_iter_hook(lambda gp: live_plot(gp, 2))
    gp.run(
        init_population_size=POPULATION_SIZE,
        init_max_depth=MAX_DEPTH,
        max_generations=MAX_GENERATIONS,
        parallelize=True,
        force_simplify=True,
        use_tqdm=True,
    )

    visualize_result(x, y, gp.best.f, block=False)



    print(f"MSE on validation set: {np.mean((gp.best.f(x_val) - y_val) ** 2):.3e}")

    try:
        gp.best.draw(block=False)
        gp.plot()
    except KeyboardInterrupt:
        pass
