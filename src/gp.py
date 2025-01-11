import logging
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from functools import cached_property

import numpy as np
from numpy.random import SFC64

from genetic_operators import (
    CollapseMutation,
    Crossover,
    ExpansionMutation,
    GeneticOperator,
    HoistMutation,
    PermutationMutation,
    PointMutation,
    SubtreeMutation,
)
from individual import Individual
from niching import Extinction
from population_selectors import (
    BalancedDeterministicSelector,
    DeterministicSelector,
    FitnessHoleSelector,
    FitnessProportionalSelector,
    TournamentSelector,
)

log = logging.getLogger(__name__)


class GP:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        seed=1,
    ):
        # normalize x and y
        # keep the normalization in memory, so that we can use it to normalize new input and de-normalize the output
        self.y_mean = np.mean(y, axis=-1)
        self.y_std = np.std(y, axis=-1)

        if self.y_std < 10:
            self.y_std = np.full_like(self.y_std, 1)
        if self.y_mean < 10:
            self.y_mean = np.full_like(self.y_mean, 0)

        # self.y_mean = np.full_like(self.y_mean, 0)
        # self.y_std = np.full_like(self.y_std, 1)

        # self.y = (y - self.y_mean) / self.y_std

        self.input_size = x.shape[0]
        self._rng = np.random.Generator(SFC64(seed))
        self._before_loop_hooks = []
        self._after_loop_hooks = []
        self._before_iter_hooks = []
        self._after_iter_hooks = []
        self._survivor_selector = None
        self._parent_selector = []
        self._fitness_function = None
        self._stop_condition = False
        self.individuals_fitness = {}
        self._exploitation_bias = 0.5
        self.reset_genetic_operators()

    @property
    def _genetic_operators(self):
        return self._exploration_operators + self._exploitation_operators

    @cached_property
    def _genetic_operators_base_weights(self):
        return np.concatenate(
            [
                np.array(self._exploration_operators_weights),
                np.array(self._exploitation_operators_weights),
            ]
        )

    @cached_property
    def __bias(self):
        return np.array(
            [1 - self._exploitation_bias] * len(self._exploration_operators_weights)
            + [self._exploitation_bias] * len(self._exploitation_operators_weights)
        )

    @cached_property
    def _genetic_operators_probs(self):
        weights = self._genetic_operators_base_weights * self.__bias
        return weights / np.sum(weights)

    def add_before_loop_hook(self, hook):
        self._before_loop_hooks.append(hook)

    def reset_before_loop_hooks(self):
        self._before_loop_hooks = []

    def add_after_loop_hook(self, hook):
        self._after_loop_hooks.append(hook)

    def reset_after_loop_hooks(self):
        self._after_loop_hooks = []

    def add_before_iter_hook(self, hook):
        self._before_iter_hooks.append(hook)

    def reset_before_iter_hooks(self):
        self._before_iter_hooks = []

    def add_after_iter_hook(self, hook):
        self._after_iter_hooks.append(hook)

    def reset_after_iter_hooks(self):
        self._after_iter_hooks = []

    def clear_cache(self, property_name):
        try:
            delattr(self, property_name)
        except AttributeError:
            pass

    def change_exploitation_bias(self, factor: float = 0.5):
        # self._exploitation_bias += (1 - self._exploitation_bias) * factor
        self._exploitation_bias = np.clip(self._exploitation_bias + factor, 0, 1)

        self.clear_cache("__bias")
        self.clear_cache("_genetic_operators_probs")

    def add_exploitation_operator(self, genetic_operator, weight):
        op = self.get_genetic_operator(genetic_operator)
        self._exploitation_operators.append(op)
        self._exploitation_operators_weights.append(weight)
        self.clear_cache("_genetic_operators_base_weights")
        self.clear_cache("__bias")
        self.clear_cache("_genetic_operators_probs")

    def add_exploration_operator(self, genetic_operator, weight):
        op = self.get_genetic_operator(genetic_operator)
        self._exploration_operators.append(op)
        self._exploration_operators_weights.append(weight)
        self.clear_cache("_genetic_operators_base_weights")
        self.clear_cache("__bias")
        self.clear_cache("_genetic_operators_probs")

    def get_genetic_operator(self, genetic_operator):
        if isinstance(genetic_operator, str):
            match genetic_operator:
                case "point":
                    genetic_operator = PointMutation
                case "subtree":
                    genetic_operator = SubtreeMutation
                case "hoist":
                    genetic_operator = HoistMutation
                case "permutation":
                    genetic_operator = PermutationMutation
                case "collapse":
                    genetic_operator = CollapseMutation
                case "expansion":
                    genetic_operator = ExpansionMutation
                case "xover" | "crossover" | "cross":
                    genetic_operator = Crossover
                case _:
                    raise ValueError("Invalid genetic operator")

        if issubclass(genetic_operator, GeneticOperator):
            genetic_operator = genetic_operator.get_new_generation
        elif not callable(genetic_operator):
            raise ValueError("Invalid genetic operator")

        return genetic_operator

    def reset_genetic_operators(self):
        self._exploration_operators = []
        self._exploitation_operators = []
        self._exploration_operators_weights = []
        self._exploitation_operators_weights = []
        self.clear_cache("_genetic_operators_base_weights")
        self.clear_cache("__bias")
        self.clear_cache("_genetic_operators_probs")

    def set_survivor_selector(self, selector):
        if isinstance(selector, str):
            match selector:
                case "deterministic":
                    selector = DeterministicSelector
                case "balanced_deterministic":
                    selector = BalancedDeterministicSelector
                case "fitness_hole":
                    selector = FitnessHoleSelector
                case "tournament":
                    selector = TournamentSelector
                case "fitness_proportional":
                    selector = FitnessProportionalSelector
                case _:
                    raise ValueError("Invalid survivor selector")
            selector = selector.select

        self._survivor_selector = selector

    def set_parent_selector(self, selector):
        if isinstance(selector, str):
            match selector:
                case "deterministic":
                    selector = DeterministicSelector
                case "tournament":
                    selector = TournamentSelector
                case "fitness_proportional":
                    selector = FitnessProportionalSelector
                case _:
                    raise ValueError("Invalid parent selector")
            selector = selector.select
        self._parent_selector = selector

    def set_fitness_function(self, hook):
        def wrapped_fitness(ind):
            if ind in self.individuals_fitness:
                return self.individuals_fitness[ind]
            fitness = hook(ind)
            self.individuals_fitness[ind] = fitness
            return fitness

        self._fitness_function = wrapped_fitness
        self._fitness_function = hook

    def add_niching_operator(self, operator):
        if isinstance(operator, str):
            match operator:
                case "extinction":
                    operator = Extinction
                case _:
                    raise ValueError("Invalid niching operator")
            operator = operator.run
        self.add_before_iter_hook(operator)

    @property
    def _tqdm_total(self):
        return self.max_generations

    @property
    def population_size(self):
        return self.population.shape[-1]

    def run(
        self,
        init_population_size: int = 10,
        init_max_depth: int = 4,
        max_generations: int = 100,
        force_simplify: bool = False,
        base_scale: float = None,
        parallelize: bool = True,
        use_tqdm: bool = True,
    ):

        self.init_population_size = init_population_size
        self.init_max_depth = init_max_depth
        self.max_generations = max_generations
        self.base_scale = base_scale
        self.force_simplify = force_simplify
        self.stale_iters = 0

        if parallelize:
            executor = ProcessPoolExecutor()
        else:
            executor = None
        self.executor = executor

        if use_tqdm:
            from tqdm import tqdm

            pbar = tqdm(total=self._tqdm_total)

        self.population = np.empty(shape=(init_population_size,), dtype=Individual)
        self.history = np.empty((max_generations, self.population_size), dtype=float)
        self.init_population(init_population_size, init_max_depth, executor)

        for hook in self._before_loop_hooks:
            hook(self)

        for self.generation in range(1, max_generations + 1):
            for hook in self._before_iter_hooks:
                hook(self)

            genetic_operator: GeneticOperator = self._rng.choice(
                self._genetic_operators, p=self._genetic_operators_probs
            )
            new_gen: np.array = genetic_operator(
                population=self.population,
                rng=self._rng,
                parent_selector=self._parent_selector,
                fitness_function=self._fitness_function,
                input_size=self.input_size,
                executor=executor,
                force_simplify=force_simplify,
            )

            assert (
                new_gen is not None
            ), f"Genetic operator {genetic_operator} did not generate any new individual"

            population = np.concatenate((self.population, new_gen), axis=0)

            self.population = self._survivor_selector(
                population=population,
                size=self.population_size,
                fitness_function=self._fitness_function,
                rng=self._rng,
            )

            self.history[self.generation - 1] = np.array(
                [self._fitness_function(ind) for ind in self.population]
            )

            if (
                self._fitness_function(self.best)
                == self.history[self.generation - 1].max()
            ):
                self.stale_iters += 1
            else:
                self.stale_iters = 0

            for hook in self._after_iter_hooks:
                hook(self)

            if use_tqdm:
                pbar.update(1)
                pbar.set_description(
                    f"Unique individuals: {len(set(self.population)):<3} - Best fitness: {self.best_fitness:.3e} - Exploitation bias: {self._exploitation_bias:.2f}"
                )

            if self._stop_condition:
                break

        if use_tqdm:
            pbar.close()

        for hook in self._after_loop_hooks:
            hook(self)

        if executor:
            executor.shutdown()

    def init_population(self, population_size, max_depth, executor: Executor = None):
        # I can't share the rng because the processes would be generating the same random numbers
        # https://stackoverflow.com/questions/72318075/is-numpy-rng-thread-safe

        if executor:
            futures = [
                executor.submit(
                    Individual,
                    initialization_method="full" if i % 2 == 0 else "grow",
                    max_depth=1 + max_depth * (i + 1) // population_size,
                    input_size=self.input_size,
                    rng=self._rng.integers(np.iinfo(np.uint32).max, dtype=np.uint32),
                    mean_y=self.y_mean,
                    std_y=self.y_std,
                )
                for i in np.arange(population_size)
            ]
            self.population[:population_size] = np.array(
                [future.result() for future in as_completed(futures)]
            )
        else:
            for i in np.arange(population_size):
                self.population[i] = Individual(
                    initialization_method="full" if i % 2 == 0 else "grow",
                    max_depth=1 + max_depth * (i + 1) // population_size,
                    input_size=self.input_size,
                    rng=self._rng,
                    mean_y=self.y_mean,
                    std_y=self.y_std,
                )

    def plot(self, block: bool = True):
        import matplotlib.pyplot as plt

        fig = plt.figure("History", figsize=(13, 13))
        for gen in np.arange(self.generation - 1):
            plt.scatter(
                x=[gen] * self.population_size,
                y=self.history[gen],
                label="Fitness",
                alpha=0.1,
                s=8,
                c="blue",
            )
        plt.plot(
            self.history[: self.generation - 1].max(axis=-1),
            label="Best fitness",
            color="red",
        )
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.yscale("log")
        plt.tight_layout()
        plt.show(block=block)

    @property
    def best(self):
        return self.population[0]

    @property
    def best_fitness(self):
        return self._fitness_function(self.best)

    @property
    def best_tree(self):
        return self.best.root

    @property
    def best_f(self):
        return self.best.f
