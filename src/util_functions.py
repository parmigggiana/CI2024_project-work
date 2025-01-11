import warnings

import matplotlib.pyplot as plt
import numpy as np

from genetic_operators import FineTuneMutation
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
        return -1

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

    if num_unique > 16:
        unique_inds = sorted(
            unique_inds, key=lambda ind: gp._fitness_function(ind), reverse=True
        )[:16]
        num_unique = 16

    cols = int(np.ceil(np.sqrt(num_unique)))
    rows = int(np.ceil(num_unique / cols))

    if not hasattr(live_plot, "fig"):
        live_plot.fig, live_plot.axes = plt.subplots(rows, cols, figsize=(16, 16))
        live_plot.axes = np.array(live_plot.axes).flatten()

    else:
        live_plot.fig.clear()
        live_plot.axes = np.array(live_plot.fig.subplots(rows, cols)).flatten()
        for ax in live_plot.axes:
            ax.clear()

    live_plot.fig.suptitle(
        f"Generation {gp.generation} - exploitation bias: {gp._exploitation_bias:.2f}"
    )

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
    DISTANCE_LABEL = "Distance from predicted value"
    # plot the best f as a plane on the set
    X0 = np.arange(x[0].min(), x[0].max(), (x[0].max() - x[0].min()) / 20)

    # Color the scatter plot by the distance to the plane
    distances = np.abs(y - f(x))

    if x.shape[0] == 1:  # Simple 2D plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(x[0], y, c=distances, cmap="magma_r")
        ax.plot(X0, f([X0]), c="r")
        fig.colorbar(scatter, ax=ax, label=DISTANCE_LABEL)
    elif x.shape[0] == 2:  # 3D plot
        fig = plt.figure()
        X1 = np.arange(x[1].min(), x[1].max(), (x[1].max() - x[1].min()) / 20)

        X = np.meshgrid(X0, X1)
        Y = np.array(f(X))

        if Y.ndim == 0:
            Y = np.full((len(X0), len(X1)), Y)

        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_zlabel("y")
        scatter = ax.scatter(x[0], x[1], y, c=distances, cmap="magma_r", alpha=1)
        ax.plot_surface(
            X[0],
            X[1],
            Y,
            alpha=0.5,
            color="r",
            rstride=1,
            cstride=1,
            edgecolor="none",
        )
        fig.colorbar(
            scatter,
            ax=ax,
            label=DISTANCE_LABEL,
            shrink=0.6,
            orientation="horizontal",
        )
    elif x.shape[0] == 3:  # 3 3D plots, one for each pair of dimensions
        fig = plt.figure()

        ax = fig.add_subplot(131, projection="3d")
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_zlabel("y")
        ax.scatter(x[0], x[1], y, c=distances, cmap="magma_r", alpha=1, s=8)
        # ax.plot_surface(X0, X1, Y, alpha=0.5, color="r", edgecolor="none")

        ax = fig.add_subplot(132, projection="3d")
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[2]")
        ax.set_zlabel("y")
        ax.scatter(x[0], x[2], y, c=distances, cmap="magma_r", alpha=1, s=8)
        # ax.plot_surface(X[0][:, 0, :], X[2][:, 0, :], Y[:, 0, :], alpha=0.5, color="r")

        ax = fig.add_subplot(133, projection="3d")
        ax.set_xlabel("x[1]")
        ax.set_ylabel("x[2]")
        ax.set_zlabel("y")
        scatter = ax.scatter(x[1], x[2], y, c=distances, cmap="magma_r", alpha=1, s=8)
        # ax.plot_surface(X[1][0, :, :], X[2][0, :, :], Y[0, :, :], alpha=0.5, color="r")

        fig.colorbar(
            scatter,
            ax=fig.axes,
            label=DISTANCE_LABEL,
            shrink=0.6,
            orientation="horizontal",
        )

    else:
        print("Cannot visualize data with more than 3 dimensions")
    plt.show(block=block)


def fine_tune_constants(gp, min_exploitation_bias, stale_window, threshold, mod):
    # if the exploitation bias is higher than min_exploitation_bias
    # and the number of generations since the last improvement is higher than min_stale_generations
    # run a FineTune operator mutations
    if gp.generation <= stale_window or gp.generation % mod != 0:
        return
    best_history = gp.history[gp.generation - stale_window - 1 : gp.generation - 1].max(
        axis=-1
    )
    if (
        best_history[-1] / best_history[0] < threshold
        and gp._exploitation_bias >= min_exploitation_bias
    ):
        new_gen = FineTuneMutation.get_new_generation(
            gp.population,
            rng=gp._rng,
            executor=gp.executor,
            force_simplify=gp.force_simplify,
        )
        population = np.concatenate((gp.population, new_gen), axis=0)

        gp.population = gp._survivor_selector(
            population=population,
            size=gp.population_size,
            fitness_function=gp._fitness_function,
            rng=gp._rng,
        )


def balance_exploitation(gp, mod=1, factor=0.01):
    if gp.generation % mod != 0:
        return

    if not hasattr(balance_exploitation, "factor"):
        balance_exploitation.factor = factor

    if gp._exploitation_bias <= 0.2 or gp._exploitation_bias >= 0.8:
        balance_exploitation.factor = -balance_exploitation.factor

    if gp.stale_iters >= mod:
        gp.change_exploitation_bias(-balance_exploitation.factor / 2)
    else:
        gp.change_exploitation_bias(balance_exploitation.factor)
