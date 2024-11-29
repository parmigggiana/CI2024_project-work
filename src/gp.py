import time

import numpy as np

from model import Node, NodeType, valid_children

function_set = [
    NodeType.ADD,
    NodeType.SUB,
    NodeType.MUL,
    NodeType.DIV,
    NodeType.SIN,
    NodeType.COS,
    NodeType.EXP,
]

terminal_set = [
    NodeType.VARIABLE,
    NodeType.CONSTANT,
]


class Individual:
    def __init__(
        self,
        root: Node = None,
        initialization_method: str = None,
        max_depth: int = None,
        x: np.ndarray = None,
        y: np.ndarray = None,
        rng: np.random.Generator = None,
    ):
        self.x = x
        self.y = y
        if rng is None:
            self.rng: np.random.Generator = np.random.rng.default_rng(
                seed=time.time_ns()
            )
        else:
            self.rng: np.random.Generator = rng

        if root is None:
            assert (
                initialization_method is not None
            ), "Either initialization method or a tree root must be specified."
            assert initialization_method in [
                "full",
                "grow",
            ], "Invalid initialization method."
            assert max_depth is not None, "Maximum depth must be specified."
            self.max_depth = max_depth
            self.initialization_method = initialization_method

            self.init_tree(max_depth=max_depth, mode=initialization_method)
        else:
            self.root = root

    @property
    def f(self):
        return self.root.f

    @property
    def nodes(self):
        return self.root.flatten

    @property
    def fitness(self, x: np.ndarray = None, y: np.ndarray = None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        mse = np.mean((self.f(x) - y) ** 2)
        if mse in [np.nan, np.inf]:
            return 0
        return -mse - self.depth

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __str__(self):
        return str(self.root)

    def __repr__(self):
        return str(self)

    def init_tree(self, max_depth: int = None, mode: str = None):
        if mode is None:
            mode = self.initialization_method

        if max_depth is None:
            max_depth = self.max_depth

        assert mode in ["full", "grow"], "Invalid initialization method."
        assert max_depth is not None, "Maximum depth must be specified."

        match mode:
            case "full":
                in_set = function_set
            case "grow":
                in_set = function_set + terminal_set

        if max_depth > 1:
            n_type = self.rng.choice(in_set)
        else:
            n_type = self.rng.choice(terminal_set)

        if n_type == NodeType.VARIABLE:
            self.root = Node(
                n_type,
                value=self.rng.integers(self.x.shape[0]),
            )
        else:
            self.root = Node(
                n_type,
                value=self.rng.random(),
            )

        nodes: list[Node] = [
            self.root,
        ]

        while nodes:
            node = nodes.pop()
            for _ in range(valid_children[node.type]):
                if node.depth == max_depth - 1:
                    n_type = self.rng.choice(terminal_set)
                else:
                    n_type = self.rng.choice(in_set)

                if n_type == NodeType.VARIABLE:
                    new_node = Node(
                        n_type,
                        value=self.rng.integers(self.x.shape[0]),
                    )
                else:
                    new_node = Node(
                        n_type,
                        value=self.rng.random(),
                    )

                node.append(new_node)
                nodes.append(new_node)
        # print(self.root)

    def clone(self):
        clone = Individual(root=self.root.clone(), x=self.x, y=self.y, rng=self.rng)
        return clone

    @property
    def depth(self):
        return max([n.depth for n in self.nodes])


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
        genetic_operator_probabilities=(0.05, 0.95),
        mutation_operators=["subtree", "point", "hoist", "permutation", "collapse"],
        min_fitness_variation_percent=0.01,
        window_size=10,
        max_generations=100,
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

        # Initialize the population
        # Half with full initialization and half with grow initialization
        # Use ramp-up max depth
        self.population = np.empty(shape=(population_size,), dtype=Individual)
        for i in np.arange(population_size, step=2):

            self.population[i] = Individual(
                initialization_method="full",
                max_depth=1 + max_depth * (i + 1) // population_size,
                x=self.x,
                y=self.y,
                rng=self.rng,
            )
            self.population[i + 1] = Individual(
                initialization_method="grow",
                max_depth=1 + max_depth * (i + 1) // population_size,
                x=self.x,
                y=self.y,
                rng=self.rng,
            )

        for _ in range(1000):
            # Choose genetic operator
            genetic_operator = self.rng.choice(
                [self.mutation, self.xover], p=genetic_operator_probabilities
            )
            new_gen = genetic_operator()
            # Select survivors
            self.population = np.concatenate((self.population, new_gen))
            # Sort the population by fitness
            self.population = sorted(self.population, reverse=True)
            self.population = self.population[:population_size]

    def xover(self) -> np.ndarray:
        # Select each individual as a parent with probability proportional to its fitness
        # The number of parents is not fixed

        p = np.array([ind.fitness for ind in self.population], dtype=np.double)
        p = p / p.sum()
        parents = self.rng.choice(
            self.population, size=self.parents, p=p, replace=False
        )
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

            if node1.parent is None:
                new_individual1.root = node2
                new_individual1.root.parent = None
                new_individual1.root.depth = 1
            else:
                node1.parent.append(node2)
                node1.parent.children.remove(node1)

            if node2.parent is None:
                new_individual2.root = node1
                new_individual2.root.parent = None
                new_individual2.root.depth = 1
            else:
                node2.parent.children.remove(node2)
                node2.parent.append(node1)

            new_gen[i] = new_individual1
            new_gen[i + 1] = new_individual2
        # print(new_gen)
        return new_gen

    def mutation(self): ...

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
