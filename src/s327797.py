import numpy as np

from gp import GP

SEED = 0
params = {
    "population_size": 400,
    "max_depth": 6,
    "reproduction_rate": 4,
    "genetic_operator_probabilities": (0.1, 0.9),
    # "mutation_operators": ["subtree", "point", "hoist", "permutation", "collapse"],
    "mutation_operators": ["point", "permutation", "hoist"],
    "mutation_rate": 0.1,
    "min_fitness_variation_percent": 0.01,
    "window_size": 5,
    "max_fitness": 1 / 1e-3,
    "max_generations": 50,
}

if __name__ == "__main__":
    problem = np.load("problem_2.npz")
    x = problem["x"]
    y = problem["y"]
    gp = GP(x, y, seed=SEED)

    gp.learn(**params)
    f = gp.best.simplify().f

    print()
    print(gp.best)
    print(f"Best is {gp.best.simplify()}")
    print(np.mean((f(x) - y) ** 2))
    print(np.mean((f(x) - y) ** 2) < 1e-3)
