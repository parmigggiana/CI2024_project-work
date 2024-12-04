from tqdm import tqdm
import numpy as np

from individual import Individual
from mutation import (
    CollapseMutation,
    HoistMutation,
    PermutationMutation,
    PointMutation,
    SubtreeMutation,
)


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
        min_fitness_variation_percent=0.01,
        window_size=10,
        max_generations=100,
        max_fitness=1 / 1e-3,
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
        self.population_size = population_size
        self.reproduction_rate = reproduction_rate
        self.min_fitness_variation_percent = min_fitness_variation_percent
        self.window_size = window_size
        self.max_generations = max_generations
        self.parents = parents
        self.mutation_rate = mutation_rate

        # Initialize the population
        # Half with full initialization and half with grow initialization
        # Use ramp-up max depth
        self.population = np.empty(shape=(population_size,), dtype=Individual)
        for i in np.arange(population_size):
            if i % 2 == 0:
                self.population[i] = Individual(
                    initialization_method="full",
                    max_depth=1 + max_depth * (i + 1) // population_size,
                    x=self.x,
                    y=self.y,
                    rng=self.rng,
                )
            else:
                self.population[i] = Individual(
                    initialization_method="grow",
                    max_depth=1 + max_depth * (i + 1) // population_size,
                    x=self.x,
                    y=self.y,
                    rng=self.rng,
                )

        for _ in tqdm(range(self.max_generations)):
            if self.best_fitness > max_fitness:
                break
            # print(self.population)
            # Choose genetic operator
            genetic_operator = self.rng.choice(
                [self.mutation, self.xover], p=genetic_operator_probabilities
            )
            new_gen = genetic_operator()
            # Select survivors
            self.population = np.concatenate((self.population, new_gen))
            # Sort the population by fitness
            self.population = sorted(self.population, reverse=True)

            # Avoid stagnation cause by over-selection
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
                self.population = self.population[:population_size]

    def xover(self) -> np.ndarray:
        # Select each individual as a parent with probability proportional to its fitness
        # The number of parents is not fixed

        p = np.array(
            [ind.fitness if not np.isnan(ind.fitness) else 0 for ind in self.population]
        )
        p = p / p.sum()
        try:
            parents = self.rng.choice(
                self.population, size=self.parents, p=p, replace=False, shuffle=False
            )
        except ValueError:
            p = np.array([ind.fitness for ind in self.population])
            print(p)

        # print(parents)
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
        # print(new_gen)
        return new_gen

    def mutation(self, strategy: str = None):
        if strategy is None:
            strategy = self.rng.choice(self.mutation_operators)

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
