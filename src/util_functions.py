import warnings

import matplotlib.pyplot as plt
import numpy as np

from gp import GP


def fitness(x, y, ind, weights: tuple):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            mse = np.mean((ind.f(x) - y) ** 2)
        except ZeroDivisionError:
            return 0
    fitness = weights[0] / mse - weights[1] * ind.depth
    if np.isnan(fitness) or np.isinf(fitness) or fitness < 0:
        return 0

    return fitness


def early_stop(gp: GP, window: int, threshold: float):
    # if the rate of change of the best fitness in the last window generations
    # is lower than threshold
    # set gp._stop_condition = True
    if gp.generation <= window:
        return
    best_history = gp.history[gp.generation - window - 1 : gp.generation - 1].max(
        axis=-1
    )
    if best_history[-1] / best_history[0] < threshold:
        gp._stop_condition = True


def live_plot(gp: GP, mod: int = 1):
    # Plot the unique individuals in subplots
    if gp.generation % mod != 0:
        return

    unique_inds = set(gp.population)
    num_unique = len(unique_inds)
    cols = int(np.ceil(np.sqrt(num_unique)))
    rows = int(np.ceil(num_unique / cols))

    if not hasattr(live_plot, "fig"):
        live_plot.fig, live_plot.axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        live_plot.axes = live_plot.axes.flatten()
    else:
        live_plot.fig.clear()
        live_plot.axes = np.array(live_plot.fig.subplots(rows, cols)).flatten()
        for ax in live_plot.axes:
            ax.clear()

    live_plot.fig.suptitle(f"Generation {gp.generation}")

    for ax in live_plot.axes[num_unique:]:
        ax.axis("off")

    for ax, ind in zip(live_plot.axes, unique_inds):
        ind.root.draw(ax=ax)
        # ax.set_title(f"Fitness: {ind.fitness:.2f}")

    plt.tight_layout()
    plt.pause(0.01)
