import time
import warnings

import numpy as np

from model import Node, NodeType, function_set, terminal_set, valid_children


class Individual:
    def __init__(
        self,
        root: Node = None,
        initialization_method: str = None,
        max_depth: int = None,
        x: np.ndarray = None,
        y: np.ndarray = None,
        w: np.ndarray = (1, 0),
        rng: np.random.Generator = None,
    ):
        self.x = x
        self.y = y
        self.w = w
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
    def fitness(
        self, x: np.ndarray = None, y: np.ndarray = None, w: tuple[float, float] = None
    ):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if w is None:
            w = self.w

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mse = np.mean((self.f(x) - y) ** 2)
        fitness = w[0] / mse - w[1] * self.depth

        if np.isnan(fitness) or np.isinf(fitness):
            return 0

        return fitness

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

    def clone(self):
        clone = Individual(root=self.root.clone(), x=self.x, y=self.y, rng=self.rng)
        return clone

    @property
    def depth(self):
        return max([n.depth for n in self.nodes])

    def simplify(self):
        if not hasattr(self, "simplified_root"):
            self.simplified_root = self.root.simplify(self.x)
        return self.simplified_root
