import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from genetic_operators import FineTuneMutation
from gp import GP
from population_selectors import DeterministicSelector


def fitness(x, y, ind):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            mse = np.mean((ind.f(x) - y) ** 2)
        except ZeroDivisionError:
            return 0
    fitness = 1 / mse / np.sqrt(ind.depth)
    if np.isnan(fitness) or np.isinf(fitness):
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


def live_plot(gp: GP, mod: int = 1, max_individuals: int = 9):
    # Plot the unique individuals in subplots
    if gp.generation % mod != 0:
        return

    unique_inds = set(gp.population)
    num_unique = len(unique_inds)

    if num_unique > max_individuals:
        unique_inds = sorted(
            unique_inds, key=lambda ind: gp._fitness_function(ind), reverse=True
        )[:max_individuals]
        num_unique = max_individuals

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
    try:
        X0 = np.arange(x[0].min(), x[0].max(), (x[0].max() - x[0].min()) / 20)
        X1 = np.arange(x[1].min(), x[1].max(), (x[1].max() - x[1].min()) / 20)
        X2 = np.arange(x[2].min(), x[2].max(), (x[2].max() - x[2].min()) / 20)
    except IndexError:
        pass

    # Color the scatter plot by the distance to the plane
    distances = np.abs(y - f(x))

    fig, ax = plt.subplots()
    if x.shape[0] == 1:  # Simple 2D plot
        scatter = ax.scatter(x[0], y, c=distances, cmap="magma_r")
        ax.plot(X0, f([X0]), c="r")
        fig.colorbar(scatter, ax=ax, label=DISTANCE_LABEL)
    elif x.shape[0] == 2:  # 3D plot
        X = np.meshgrid(X0, X1)
        Y = np.array(f(X))

        if Y.ndim == 0:
            Y = np.full((len(X0), len(X1)), Y)
        ax.axis("off")
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
    # elif x.shape[0] == 3:  # One 3D plot with a slider for the 3rd input dimension
    #     ax.axis("off")
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.set_xlabel("x[0]")
    #     ax.set_ylabel("x[1]")
    #     ax.set_zlabel("y")
    #     ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    #     slider = Slider(
    #         ax=ax_slider,
    #         label="X2",
    #         valmin=X2.min(),
    #         valmax=X2.max(),
    #         valinit=X2.mean(),
    #     )

    #     X = np.meshgrid(X0, X1, X2)
    #     Y = f(X)

    #     def update(val):
    #         ax.clear()
    #         ax.set_xlabel("x[0]")
    #         ax.set_ylabel("x[1]")
    #         ax.set_zlabel("y")
    #         idx = np.argmin(np.abs(X2 - val))

    #         distances = np.abs(y - f([x[0], x[1], np.full_like(x[0], val)]))
    #         scatter = ax.scatter(x[0], x[1], y, c=distances, cmap="magma_r", alpha=1)

    #         ax.plot_surface(
    #             X[0][:, :, idx],
    #             X[1][:, :, idx],
    #             Y[:, :, idx],
    #             alpha=0.5,
    #             color="r",
    #             edgecolor="none",
    #         )
    #         fig.canvas.draw_idle()

    #     slider.on_changed(update)
    #     update(0)

    elif x.shape[0] == 3:  # 3 3D plots, one for each pair of dimensions
        ax.axis("off")
        ax = fig.add_subplot(131, projection="3d")
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_zlabel("y")
        ax.scatter(x[0], x[1], y, c=distances, cmap="magma_r", alpha=1, s=8)

        X = np.meshgrid(X0, X1)
        dum = np.zeros_like(X[0])
        X = np.stack((X[0], X[1], dum), axis=0)
        Y1 = f(X)
        ax.plot_surface(X[0], X[1], Y1, alpha=0.5, color="r", edgecolor="none")

        ax = fig.add_subplot(132, projection="3d")
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[2]")
        ax.set_zlabel("y")
        ax.scatter(x[0], x[2], y, c=distances, cmap="magma_r", alpha=1, s=8)
        X = np.meshgrid(X0, X2)
        X = np.stack((X[0], dum, X[1]), axis=0)
        Y2 = f(X)
        ax.plot_surface(X[0], X[2], Y2, alpha=0.5, color="r")

        ax = fig.add_subplot(133, projection="3d")
        ax.set_xlabel("x[1]")
        ax.set_ylabel("x[2]")
        ax.set_zlabel("y")
        scatter = ax.scatter(x[1], x[2], y, c=distances, cmap="magma_r", alpha=1, s=8)
        X = np.meshgrid(X1, X2)
        X = np.stack((dum, X[0], X[1]), axis=0)
        Y3 = f(X)
        ax.plot_surface(X[1], X[2], Y3, alpha=0.5, color="r")

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


def fine_tune_constants(gp: GP):
    # Iterate until there's no improvement in the best individual
    print("Starting fine tuning")
    best_fitness = gp._fitness_function(gp.best)
    while True:
        new_gen = FineTuneMutation.get_new_generation(
            gp.population,
            rng=gp._rng,
            executor=gp.executor,
            force_simplify=gp.force_simplify,
        )
        population = np.concatenate((gp.population, new_gen), axis=0)
        gp.population = DeterministicSelector.select(
            population=population,
            size=gp.population_size,
            fitness_function=gp._fitness_function,
        )
        new_best_fitness = gp._fitness_function(gp.best)
        if new_best_fitness / best_fitness <= 1 + 1e-5:
            break


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
