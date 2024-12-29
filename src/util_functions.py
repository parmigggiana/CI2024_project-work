import warnings

import matplotlib.pyplot as plt
import numpy as np

from gp import GP


def fitness(x, y, ind):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            mse = np.mean((ind.f(x) - y) ** 2)
        except ZeroDivisionError:
            return 0
    fitness = 1 / mse / np.sqrt(ind.depth)
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
        live_plot.axes = np.array(live_plot.axes).flatten()

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

def visualize_data(x, y):
    fig, ax = plt.subplots()

    if x.shape[0] == 1:
        ax.scatter(x[0], y)
    elif x.shape[0] == 2:
        # 3D plot
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x[0], x[1], y)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_zlabel("y")
    else:
        raise ValueError("x must have 1 or 2 rows")
    plt.show(block=False)
    return fig, ax

def visualize_result(x, y, f, block=None):
    # plot the best f as a plane on the set
    fig, ax = plt.subplots()

    X0 = np.arange(x[0].min(), x[0].max(), 0.1)

    # Color the scatter plot by the distance to the plane
    distances = np.abs(y - f(x))
    if x.shape[0] == 1:
        scatter = ax.scatter(x[0], y, c=distances, cmap='magma_r')
        ax.plot(X0, f([X0]), c="r")
    elif x.shape[0] == 2:
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(x[0], x[1], y, c=distances, cmap='magma_r')

        X1 = np.arange(x[1].min(), x[1].max(), 0.1)
        X = np.meshgrid(X0, X1)
        ax.plot_surface(X[0], X[1], f(X), alpha=0.5, color='r')
    fig.colorbar(scatter, ax=ax, label='Distance from predicted value')
    plt.show(block=block)

def change_exploitation_bias(gp: GP, mod: int = 1, factor: float = 1.01):
    if gp.generation % mod != 0:
        return

    gp._exploitation_bias *= factor