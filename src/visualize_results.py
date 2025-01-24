import sys

import numpy as np
import importlib.util
from util_functions import visualize_result
import os

PROBLEM = sys.argv[1]

x = np.load(f"data/problem_{PROBLEM}.npz")["x"]
y = np.load(f"data/problem_{PROBLEM}.npz")["y"]

# Load f from s327797.py
spec = importlib.util.spec_from_file_location(
    "f", os.path.join(os.path.dirname(__file__), "../s327797.py")
)
s327797 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s327797)
f = getattr(s327797, f"f{PROBLEM}")

mse = np.mean((y - f(x)) ** 2)
nrmse = np.sqrt(np.mean((y - f(x)) ** 2)) / (y.max() - y.min())
print(f"MSE: {mse:.3e}")
print(f"NRMSE: {nrmse:.3e}")
visualize_result(x, y, f, block=True)
