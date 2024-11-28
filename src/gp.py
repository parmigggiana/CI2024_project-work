import time
from typing import Callable

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

            if initialization_method == "full":
                self.init_full()
            elif initialization_method == "grow":
                self.init_grow()
        else:
            self.root = root
            self.f = self.root.f


        if x is not None and y is not None:
            self.evaluate()

    def evaluate(self, x: np.ndarray = None, y: np.ndarray = None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        self.fitness = np.mean((self.root.f(x) - y) ** 2)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __str__(self):
        return str(self.root)

    def __repr__(self):
        return str(self)

    def init_full(self, max_depth: int = None):
        if max_depth is None:
            max_depth = self.max_depth

        if max_depth > 1:
            n_type = self.rng.choice(list(NodeType))
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

        nodes = [self.root]
        while nodes:
            node = nodes.pop()
            for _ in range(valid_children[node.type]):
                if node.depth == max_depth - 2:
                    n_type = self.rng.choice(terminal_set)
                    if n_type == NodeType.VARIABLE:
                        new_node = Node(
                            n_type,
                            value=self.rng.integers(self.x.shape[0]),
                            depth=node.depth + 1,
                        )
                    else:
                        new_node = Node(
                            n_type,
                            value=self.rng.random(),
                            depth=node.depth + 1,
                        )
                else:
                    new_node = Node(self.rng.choice(function_set), depth=node.depth + 1)
                node.children.append(new_node)
                nodes.append(new_node)
        self.f = self.root.f


    def init_grow(self, max_depth: int = None):
        if max_depth is None:
            max_depth = self.max_depth

        if max_depth > 1:
            n_type = self.rng.choice(list(NodeType))
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
        nodes = [self.root]

        while nodes:
            node = nodes.pop()
            for _ in range(valid_children[node.type]):
                if node.depth == max_depth - 2:
                    n_type = self.rng.choice(terminal_set)
                else:
                    n_type = self.rng.choice(list(NodeType))

                if n_type == NodeType.VARIABLE:
                    new_node = Node(
                        n_type,
                        value=self.rng.integers(self.x.shape[0]),
                        depth=node.depth + 1,
                    )
                else:
                    new_node = Node(
                        n_type,
                        value=self.rng.random(),
                        depth=node.depth + 1,
                    )
                node.children.append(new_node)
                nodes.append(new_node)
        self.f = self.root.f


class GP:
    def __init__(self, x: np.ndarray, y: np.ndarray, seed=1):
        self.x = x
        self.y = y
        self.cardinality = x.shape[0]
        self.rng = np.random.default_rng(seed)

    def learn(
        self, population_size=1, max_depth=1
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Randomly generate a tree and learn the best tree to fit the data.
        """
        # Initialize the population
        # Half with full initialization and half with grow initialization
        # Evaluate the fitness of each individual
        population = np.empty(shape=(population_size,), dtype=Individual)
        for i in range(population_size):
            if i % 2:
                population[i] = Individual(
                    initialization_method="full",
                    max_depth=max_depth,
                    x=self.x,
                    y=self.y,
                    rng=self.rng,
                )
            else:
                population[i] = Individual(
                    initialization_method="grow",
                    max_depth=max_depth,
                    x=self.x,
                    y=self.y,
                    rng=self.rng,
                )

        # Sort the population by fitness
        population.sort()
        best = population[0]

        self.best = best

    def random_node(self):
        # Randomly generate a node
        # Randomly choose a node type
        # If the node type is a variable, randomly choose a variable index
        # If the node type is a constant, randomly choose a constant value
        # Use weights to bias the selection of the node type
        # Prefer nodes with lower branching factor
        # p = 3 - np.array(list(valid_children.values()))
        # p = p / p.sum()
        p = [0.35, 0.35, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]
        p = np.array(p) / np.sum(p)
        node_type = self.rng.choice(list(NodeType), p=p)
        if node_type == NodeType.VARIABLE:
            return Node(node_type, self.rng.integers(0, self.cardinality))
        elif node_type == NodeType.CONSTANT:
            return Node(node_type, self.rng.random())
        else:
            return Node(node_type)
