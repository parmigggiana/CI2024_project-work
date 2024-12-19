import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

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
    DeterministicSelector,
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
        self.x = x
        self.y = y
        self.input_size = x.shape[0]
        self._rng = np.random.default_rng(seed)
        self._before_loop_hooks = []
        self._after_loop_hooks = []
        self._before_iter_hooks = []
        self._after_iter_hooks = []
        self._genetic_operators = []
        self._genetic_operators_probs = []
        self._survivor_selector = None
        self._parent_selector = []
        self._fitness_function = None
        self._stop_condition = False
        self.individuals_fitness = {}

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

    def add_genetic_operator(self, genetic_operator, probability):
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
            genetic_operator = genetic_operator.get_new_generation

        self._genetic_operators.append(genetic_operator)
        self._genetic_operators_probs.append(probability)

    def reset_genetic_operators(self):
        self._genetic_operators = []
        self._genetic_operators_probs = []

    def set_survivor_selector(self, selector):
        if isinstance(selector, str):
            match selector:
                case "deterministic":
                    selector = DeterministicSelector
                case "tournament":
                    selector = self._tournament_survivor_selection
                case "fitness_proportional":
                    selector = self._fitness_proportional_survivor_selection
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
        parallelize: bool = True,
        force_simplify: bool = False,
        use_tqdm: bool = True,
    ):

        self.init_population_size = init_population_size
        self.init_max_depth = init_max_depth
        self.max_generations = max_generations
        self.population = np.empty(shape=(init_population_size,), dtype=Individual)
        self.init_population(init_population_size, init_max_depth, parallelize)
        self.history = np.empty((max_generations, self.population_size), dtype=float)
        self.parallelize = parallelize

        if use_tqdm:
            from tqdm import tqdm

        for hook in self._before_loop_hooks:
            hook(self)

        if use_tqdm:
            pbar = tqdm(total=self._tqdm_total)

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
                parallelize=parallelize,
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
            )
            # ic([self._fitness_function(ind) for ind in self.population])
            for hook in self._after_iter_hooks:
                hook(self)

            self.history[self.generation - 1] = np.array(
                [self._fitness_function(ind) for ind in self.population]
            )

            if use_tqdm:
                pbar.update(1)
                pbar.set_description(
                    f"Unique individuals: {len(set(self.population)):<3} - Best fitness: {self.best_fitness:.3e}"
                )

            if self._stop_condition:
                break

        if use_tqdm:
            pbar.close()

        for hook in self._after_loop_hooks:
            hook(self)

    def init_population(self, population_size, max_depth, parallelize=True):
        if parallelize:
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        Individual,
                        initialization_method="full" if i % 2 == 0 else "grow",
                        max_depth=1 + max_depth * (i + 1) // population_size,
                        input_size=self.input_size,
                        rng=np.random.default_rng(
                            self._rng.integers(0, np.iinfo(np.int32).max)
                        ),
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
                )

    def plot(self, block: bool = True):
        import matplotlib.pyplot as plt

        plt.figure()
        for gen in np.arange(self.generation - 1):
            plt.scatter(
                x=[gen] * self.population_size,
                y=self.history[gen],
                label="Fitness",
                alpha=0.2,
                c="blue",
            )
        plt.plot(
            self.history[: self.generation - 1].max(axis=-1),
            label="Best fitness",
            color="red",
        )
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
