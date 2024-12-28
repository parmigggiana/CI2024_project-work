import time

import numpy as np

from model import Node, NodeType, function_set, terminal_set, valid_children


class Individual:
    def __init__(
        self,
        root: Node = None,
        initialization_method: str = None,
        max_depth: int = None,
        input_size: int = None,
        rng: np.random.Generator = None,
    ):
        self.input_size = input_size
        if rng is None:
            self.rng: np.random.Generator = np.random.default_rng(seed=time.time_ns())
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

    def __str__(self):
        return str(self.root)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.root == other.root

    def __hash__(self):
        return hash(self.root)

    def get_random_node(self, node_type):
        if node_type == NodeType.VARIABLE:
            return Node(
                node_type,
                value=self.rng.integers(self.input_size),
            )
        else:
            return Node(
                node_type,
                value=self.rng.random(),
            )

    def init_tree(self, max_depth: int = None, mode: str = None):
        if mode is None:
            mode = self.initialization_method

        if max_depth is None:
            max_depth = self.max_depth

        assert mode in ["full", "grow"], "Invalid initialization method."
        assert max_depth is not None, "Maximum depth must be specified."

        if mode == "full":
            in_set = function_set
        elif mode == "grow":
            in_set = function_set + terminal_set

        if max_depth == 1:
            n_type = self.rng.choice(terminal_set)
        else:
            n_type = self.rng.choice(in_set)

        self.root = self.get_random_node(n_type)
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

                new_node = self.get_random_node(n_type)

                node.append(new_node)
                nodes.append(new_node)

    def clone(self):
        clone = Individual(root=self.root.clone(), rng=self.rng)
        return clone

    @property
    def depth(self):
        return max([n.depth for n in self.nodes])

    def simplify(self):
        if not hasattr(self, "simplified_root") or not self.simplified_root:
            self.root = self.root.simplify()
            self.simplified_root = True

    def draw(self, block=True, ax=None):
        self.root.draw(block=block, ax=ax)
