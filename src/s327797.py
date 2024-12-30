import numpy as np


def f0(x):
    """
    SEED = 42
    POPULATION_SIZE = 500
    MAX_DEPTH = 5
    MAX_GENERATIONS = 2000
    EARLY_STOP_WINDOW_SIZE = 600
    """
    return x[0] + (0.203 * np.sin(x[1]))


def f1(x):
    return np.sin(x[0])
