import logging
import time
from abc import ABC, abstractmethod

import numpy as np

from individual import Individual
from model import Node, NodeType, reverse_valid_children, valid_children

log = logging.getLogger(__name__)


class GeneticOperator(ABC):
    @classmethod
    @abstractmethod
    def get_new_generation(cls, population, *, rng=None, **kwargs):
        raise NotImplementedError(f"{cls.__name__} is not yet implemented.")


class Crossover(GeneticOperator):
    @classmethod
    def get_new_generation(
        cls,
        population,
        parent_selector,
        *,
        rng=None,
        reproduction_rate=2,
        **kwargs,
    ):
        if rng is None:
            rng = np.random.default_rng(time.time_ns())

        cls.rng = rng

        parents = parent_selector(
            population, rng=cls.rng, fitness_function=kwargs["fitness_function"]
        )

        new_gen = np.empty(
            shape=(population.shape[-1] * reproduction_rate,), dtype=Individual
        )
        for i in np.arange(population.shape[-1] * reproduction_rate, step=2):
            try:
                parent1, parent2 = rng.choice(parents, size=2, replace=False)
            except ValueError:
                log.error("Not enough parents to perform crossover.")
                log.error(f"Parents selector: {parent_selector}")
                new_gen = np.full_like(new_gen, parents[0])
                break

            new_individual1 = parent1.clone()
            new_individual2 = parent2.clone()

            node1 = rng.choice(new_individual1.nodes)
            node2 = rng.choice(new_individual2.nodes)
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

    @classmethod
    def _select_parents_fitness_proportional(cls, population, fitness_function):
        p = np.array([fitness_function(individual) for individual in population])
        p -= (
            np.min(p) - 1e-9
        )  # if all fitness are the same, the probability would be 0 without 1e-9
        p /= p.sum()
        p *= len(p)

        parents = population[cls.rng.random(size=population.shape[-1]) < p]

        return parents


class Mutation(GeneticOperator):
    @classmethod
    def get_new_generation(cls, population, *, mutation_rate=1, rng=None, **kwargs):
        if rng is None:
            rng = np.random.default_rng(time.time_ns())

        cls.rng = rng
        new_gen = population.copy()
        for individual in new_gen:
            if rng.random() < mutation_rate:
                cls.mutate(individual, **kwargs)

        return new_gen

    @classmethod
    def mutate(cls, individual: Individual, **kwargs):
        raise NotImplementedError(f"{cls.__name__} is not yet implemented.")


class SubtreeMutation(Mutation): ...


class PointMutation(Mutation):
    @classmethod
    def mutate(cls, individual, input_size: int, **kwargs):
        node = cls.rng.choice(individual.nodes)
        new_type = cls.rng.choice(reverse_valid_children[valid_children[node.type]])
        node.type = new_type
        if node.type == NodeType.CONSTANT:
            node.value = cls.rng.random()
        elif node.type == NodeType.VARIABLE:
            node.value = cls.rng.integers(input_size)


class PermutationMutation(Mutation):
    @classmethod
    def mutate(cls, individual, **kwargs):
        valid_nodes = [
            node for node in individual.nodes if valid_children[node.type] == 2
        ]
        if valid_nodes:
            node = cls.rng.choice(valid_nodes)
            while valid_children[node.type] != 2:
                node = cls.rng.choice(valid_nodes)
            node.children[0], node.children[1] = node.children[1], node.children[0]


class HoistMutation(Mutation):
    @classmethod
    def mutate(cls, individual, **kwargs):
        valid_nodes = [
            node for node in individual.nodes if valid_children[node.type] == 2
        ]
        if len(valid_nodes) > 1:
            node = cls.rng.choice(valid_nodes)
            while node.parent is None:
                node = cls.rng.choice(valid_nodes)
            individual.root = Node(node.type, node.value)
            individual.root._children = node.children
            # Set depths
            nodes = [
                individual.root,
            ]
            while nodes:
                node = nodes.pop()
                for child in node._children:
                    child.depth = node.depth + 1
                    nodes.append(child)


class CollapseMutation(Mutation): ...


class ExpansionMutation(Mutation): ...
