from typing import Callable

import numpy as np

from gp import GP
from model import Node, NodeType

if __name__ == "__main__":
    problem = np.load("problem_0.npz")
    x = problem["x"]
    y = problem["y"]
    gp = GP(x, y, seed=0)
    gp.learn(population_size=10000, max_depth=3)
    f = gp.best.f
    # print(f(x))
    # print(y)
    print()
    print(f"Best is {gp.best}")
    print(np.mean((f(x) - y) ** 2))
    print(np.mean((f(x) - y) ** 2) < 1e-3)
