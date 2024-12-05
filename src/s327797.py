import numpy as np

from gp import GP

SEED = None
PROBLEM = 0
params = {
    "population_size": 80,
    "max_depth": 6,
    "reproduction_rate": 4,
    "genetic_operator_probabilities": (0.1, 0.9),
    "mutation_operators": [  # "subtree", "point", "hoist", "permutation", "collapse"
        "point",
        "hoist",
        "permutation",
    ],
    "mutation_probabilities": (0.1, 0.6, 0.3),
    "fitness_weights": (0.9, 0.1),
    "mutation_rate": 0.8,
    "min_fitness_variation_percent": 0.05,
    "window_size": 200,
    "max_fitness": 1 / 1e-4,
    "max_generations": 1000,
}

if __name__ == "__main__":
    problem = np.load(f"tests/problem_{PROBLEM}.npz")
    x = problem["x"]
    y = problem["y"]
    gp = GP(x, y, seed=SEED)

    gp.learn(**params)
    f = gp.best.simplify().f

    print()
    # print(gp.best)
    print(f"Best is {gp.best.simplify()}")
    print(f"Found in {gp.generations} generations")
    print(f"MSE on training set: {np.mean((f(x) - y) ** 2):.3e}")

    validation = np.load(f"tests/validation_{PROBLEM}.npz")
    x_val = validation["x"]
    y_val = validation["y"]
    print(f"MSE on validation set: {np.mean((f(x_val) - y_val) ** 2):.3e}")
    try:
        gp.best.simplify().draw()
    except KeyboardInterrupt:
        pass
