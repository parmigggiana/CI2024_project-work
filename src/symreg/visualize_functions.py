import importlib.util

import numpy as np

from symreg.util_functions import visualize_result


def main(args):
    problem = args.problem

    x = np.load(f"data/problem_{problem}.npz")["x"]
    y = np.load(f"data/problem_{problem}.npz")["y"]

    # Load f from s327797.py
    spec = importlib.util.spec_from_file_location("f", args.file)

    s327797 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s327797)
    f = getattr(s327797, f"f{problem}")

    mse = np.mean((y - f(x)) ** 2)
    nrmse = np.sqrt(np.mean((y - f(x)) ** 2)) / (y.max() - y.min())
    print(f"MSE: {mse:.3e}")
    print(f"NRMSE: {nrmse:.3e}")
    visualize_result(x, y, f, block=True)
