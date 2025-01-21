import importlib
import pickle
import sys

import numpy as np

from util_functions import visualize_result

PROBLEM = sys.argv[1]

x = np.load(f"data/problem_{PROBLEM}.npz")["x"]
y = np.load(f"data/problem_{PROBLEM}.npz")["y"]
# with open(f"results/problem_{PROBLEM}", "rb") as fs:
#     f = pickle.load(fs)
mod = importlib.import_module(f"s327797")
f = mod.__getattribute__(f"f{PROBLEM}")

# f = lambda x: x[0] + np.sin(x[1]) - 0.8 * np.sin(x[1])
print(f"MSE: {np.mean((y - f(x)) ** 2):.3e}")
nrmse = np.sqrt(np.mean((y - f(x)) ** 2)) / (y.max() - y.min())
print(f"NRMSE: {nrmse:.3e}")
visualize_result(x, y, f, block=True)
