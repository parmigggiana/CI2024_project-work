import time
from zoneinfo import available_timezones

from matplotlib.style import available
import numpy as np

from model import Node, NodeType, reverse_valid_children, valid_children
from individual import Individual


class MutationStrategy:
    @classmethod
    def mutate_population(cls, population, mutation_rate, rng=None, **kwargs):
        if rng is None:
            rng = np.random.default_rng(time.time_ns())

        cls.rng = rng
        for individual in population:
            if rng.random() < mutation_rate:
                cls.mutate(individual, **kwargs)

    @classmethod
    def mutate(cls, individual: Individual, **kwargs):
        raise NotImplementedError(f"{cls.__name__} is not yet implemented.")


class SubtreeMutation(MutationStrategy): ...


class PointMutation(MutationStrategy):
    @classmethod
    def mutate(cls, individual, input_size: int):
        node = cls.rng.choice(individual.nodes)
        new_type = cls.rng.choice(reverse_valid_children[valid_children[node.type]])
        node.type = new_type
        if node.type == NodeType.CONSTANT:
            node.value = cls.rng.random()
        elif node.type == NodeType.VARIABLE:
            node.value = cls.rng.integers(input_size)


class PermutationMutation(MutationStrategy):
    @classmethod
    def mutate(cls, individual):
        valid_nodes = [
            node for node in individual.nodes if valid_children[node.type] == 2
        ]
        if valid_nodes:
            node = cls.rng.choice(valid_nodes)
            while not valid_children[node.type] == 2:
                node = cls.rng.choice(valid_nodes)
            node.children[0], node.children[1] = node.children[1], node.children[0]


class HoistMutation(MutationStrategy):
    @classmethod
    def mutate(cls, individual: Individual):
        valid_nodes = [
            node for node in individual.nodes if valid_children[node.type] == 2
        ]
        if valid_nodes:
            node = cls.rng.choice(valid_nodes)
            while node.parent is None:
                node = cls.rng.choice(valid_nodes)
            individual.root = Node(node.type, node.value, children=node.children)
            # Set depths
            nodes = [
                individual.root,
            ]
            while nodes:
                node = nodes.pop()
                for child in node.children:
                    child.depth = node.depth + 1
                    nodes.append(child)


class CollapseMutation(MutationStrategy): ...
