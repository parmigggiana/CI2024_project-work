import numpy as np

from gp import GP

SEED = 0
params = {
    "population_size": 1000,
    "max_depth": 5,
    "reproduction_rate": 2,
    "genetic_operator_probabilities": (0.0, 1),
    "mutation_operators": ["subtree", "point", "hoist", "permutation", "collapse"],
    "min_fitness_variation_percent": 0.01,
    "window_size": 5,
    "max_generations": 100,
}

if __name__ == "__main__":
    problem = np.load("problem_0.npz")
    x = problem["x"]
    y = problem["y"]
    gp = GP(x, y, seed=SEED)

    gp.learn(**params)
    f = gp.best.f
    # print(f(x))
    # print(y)
    print()
    print(f"Best is {gp.best}")
    print(np.mean((f(x) - y) ** 2))
    print(np.mean((f(x) - y) ** 2) < 1e-3)
