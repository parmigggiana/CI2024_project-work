import pickle
import sys

import numpy as np

from individual import Individual
from util_functions import visualize_result

PROBLEM = sys.argv[1]

with open(f"results/problem_{PROBLEM}.txt", "rb") as f:
    ind: Individual = pickle.load(f).clone()

x = np.load(f"data/problem_{PROBLEM}.npz")["x"]
y = np.load(f"data/problem_{PROBLEM}.npz")["y"]

# def f(x):
#     return (1 + 2 * np.cos(x[1])) * np.pi - x[0] / 2 / np.pi
f = ind.f


visualize_result(x, y, f, block=True)
