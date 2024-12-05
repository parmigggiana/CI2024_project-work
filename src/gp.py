import numpy as np
from tqdm import tqdm

from individual import Individual
from mutation import (CollapseMutation, HoistMutation, PermutationMutation,
                      PointMutation, SubtreeMutation)


class GP:
    def __init__(self, x: np.ndarray, y: np.ndarray, seed=1):
        self.x = x
        self.y = y
        self.cardinality = x.shape[0]
        self.rng = np.random.default_rng(seed)

    def learn(
        self,
        population_size=1,
        max_depth=1,
        parents=2,
        reproduction_rate=2,
        mutation_rate=0.05,
        genetic_operator_probabilities=(0.05, 0.95),
        mutation_operators=["subtree", "point", "hoist", "permutation", "collapse"],
        mutation_probabilities=None,
        min_fitness_variation_percent=0.01,
        window_size=10,
        max_generations=100,
        max_fitness=1 / 1e-3,
        fitness_weights=(1, 0),
    ) -> None:
        """Randomly generate a tree and learn the best tree to fit the data.



        Args:
            population_size (int, optional): The size of the population at init and after survivor selection. Defaults to 1.
            max_depth (int, optional): The maximum depth of the initial population's trees. Defaults to 1.
            p (tuple, optional): Probabilities for selecting the genetic operator. (mutation, crossover). Defaults to (0.05, 0.95).

        Returns:
            None: This method sets self.best with the best-fitness Individual after the process
        """
        self.mutation_operators = mutation_operators
        self.mutation_probabilities = mutation_probabilities
        self.population_size = population_size
        self.reproduction_rate = reproduction_rate
        self.min_fitness_variation_percent = min_fitness_variation_percent
        self.window_size = window_size
        self.max_generations = max_generations
        self.parents = parents
        self.mutation_rate = mutation_rate
        self.fitness_weights = fitness_weights

        # Initialize the population
        # Half with full initialization and half with grow initialization
        # Use ramp-up max depth
        self.population = np.empty(shape=(population_size,), dtype=Individual)
        self.init_population(population_size, max_depth)
        self.resets = 0
        self.history = np.array([])

        with tqdm(
            desc=f"Best fitness: {self.best_fitness:<10.2f} Different trees: {len(set(map(lambda x: x.root,self.population))):<3}",
            total=self.max_generations,
        ) as pbar:
            for gen in range(1, self.max_generations + 1):
                if self.best_fitness > max_fitness:
                    break
                self.generations = gen

                # Choose genetic operator
                genetic_operator = self.rng.choice(
                    [self.mutation, self.xover], p=genetic_operator_probabilities
                )
                new_gen = genetic_operator()

                # Select survivors
                self.population = np.concatenate((self.population, new_gen))
                self.population = sorted(self.population, reverse=True)

                # Avoid stagnation cause by over-selection
                # TODO Implement a better way to avoid stagnation
                if self.population_size >= 1000:
                    group1 = self.population[: 320 * self.reproduction_rate]
                    group2 = self.population[320 * self.reproduction_rate :]
                    self.population = np.concatenate(
                        (
                            group1[: int(self.population_size * 0.8)],
                            group2[: int(self.population_size * 0.2)],
                        )
                    )
                else:
                    group1 = self.population[:population_size]
                    group2 = self.population[population_size:]
                    self.population = np.concatenate(
                        (
                            group1[: int(self.population_size * 0.8)],
                            group2[: int(self.population_size * 0.2)],
                        )
                    )
                self.different_trees = len(set(map(lambda x: x.root, self.population)))
                # Visualize the best tree and the number of different trees

                self.history = np.append(self.history, self.best_fitness)

                if len(self.history) > self.window_size:
                    self.history = self.history[1:]
                    if (self.history[-1] - self.history[0]) / len(
                        self.history
                    ) / self.history[0] < self.min_fitness_variation_percent:
                        break  # can't improve anymore
                elif len(self.history) > (self.window_size // 2):
                    if (self.history[-1] - self.history[self.window_size // 2]) / len(
                        self.history
                    ) / self.history[0] < 2 * self.min_fitness_variation_percent:
                        self.population[-self.different_trees :] = [
                            Individual(
                                root,
                                x=self.x,
                                y=self.y,
                                w=self.fitness_weights,
                                rng=self.rng,
                            )
                            for root in set(map(lambda x: x.root, self.population))
                        ]
                        self.init_population(
                            population_size - self.different_trees, max_depth
                        )

                pbar.set_description(
                    f"Best fitness: {self.best_fitness:<10.2f} Different trees: {self.different_trees:<3}",
                    refresh=False,
                )
                pbar.update(1)

                # Reset while keeping best to avoid stagnation
                if self.different_trees == 1:
                    self.population[-1] = self.population[0]
                    self.init_population(population_size - 1, max_depth)

    def init_population(self, population_size, max_depth):
        for i in np.arange(population_size):
            if i % 2 == 0:
                self.population[i] = Individual(
                    initialization_method="full",
                    max_depth=1 + max_depth * (i + 1) // population_size,
                    x=self.x,
                    y=self.y,
                    w=self.fitness_weights,
                    rng=self.rng,
                )
            else:
                self.population[i] = Individual(
                    initialization_method="grow",
                    max_depth=1 + max_depth * (i + 1) // population_size,
                    x=self.x,
                    y=self.y,
                    w=self.fitness_weights,
                    rng=self.rng,
                )

    def xover(self) -> np.ndarray:
        # Select each individual as a parent with probability proportional to its fitness
        # The number of parents is not fixed

        p = np.array([ind.fitness for ind in self.population])
        p -= (
            np.min(p) - 1e-9
        )  # if all fitness are the same, the probability would be 0 without 1e-9
        p = p / p.sum()

        parents = self.rng.choice(
            self.population, size=self.parents, p=p, replace=False, shuffle=False
        )

        new_gen = np.empty(
            shape=(self.population_size * self.reproduction_rate,), dtype=Individual
        )
        for i in np.arange(self.population_size * self.reproduction_rate, step=2):
            parent1, parent2 = self.rng.choice(parents, size=2, replace=False)

            new_individual1 = parent1.clone()
            new_individual2 = parent2.clone()

            node1 = self.rng.choice(new_individual1.nodes)
            node2 = self.rng.choice(new_individual2.nodes)
            node2_p = node2.parent

            if node1.parent is None:
                new_individual1.root = node2
                new_individual1.root.parent = None
                new_individual1.root.depth = 1
            else:
                node1.parent.children.remove(node1)
                node1.parent.append(node2)  # This changes node2.parent and node2.depth

            if node2_p is None:
                new_individual2.root = node1
                new_individual2.root.parent = None
                new_individual2.root.depth = 1
            else:
                node2_p.children.remove(node2)
                node2_p.append(node1)

            new_gen[i] = new_individual1
            new_gen[i + 1] = new_individual2
        return new_gen

    def mutation(self, strategy: str = None, p=None) -> np.ndarray:
        if p is None:
            p = self.mutation_probabilities
        if strategy is None:
            strategy = self.rng.choice(self.mutation_operators, p=p)

        new_gen = list(map(lambda ind: ind.clone(), self.population))

        match strategy:
            case "point":
                PointMutation.mutate_population(
                    new_gen,
                    self.mutation_rate,
                    rng=self.rng,
                    input_size=self.x.shape[0],
                )
            case "subtree":
                SubtreeMutation.mutate_population(
                    new_gen,
                    self.mutation_rate,
                    rng=self.rng,
                )

            case "hoist":
                HoistMutation.mutate_population(
                    new_gen,
                    self.mutation_rate,
                    rng=self.rng,
                )

            case "permutation":
                PermutationMutation.mutate_population(
                    new_gen,
                    self.mutation_rate,
                    rng=self.rng,
                )

            case "collapse":
                CollapseMutation.mutate_population(
                    new_gen,
                    self.mutation_rate,
                    rng=self.rng,
                )

            case _:
                raise ValueError("Invalid mutation strategy")
        return new_gen

    @property
    def best(self):
        return self.population[0]

    @property
    def best_fitness(self):
        return self.best.fitness

    @property
    def best_tree(self):
        return self.best.root

    @property
    def best_f(self):
        return self.best.f
