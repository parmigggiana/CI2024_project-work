import logging
from abc import ABC, abstractmethod
from concurrent.futures import Executor, as_completed

import numpy as np
from numpy.random import SFC64

from symreg.individual import Individual
from symreg.model import (Node, NodeType, reverse_valid_children, terminal_set,
                          valid_children)

log = logging.getLogger(__name__)


class GeneticOperator(ABC):
    @classmethod
    @abstractmethod
    def get_new_generation(
        cls, population, *, rng=None, force_simplify: bool = False, **kwargs
    ):
        raise NotImplementedError(f"{cls.__name__} is not yet implemented.")


class Crossover(GeneticOperator):
    @classmethod
    def get_new_generation(
        cls,
        population,
        parent_selector,
        *,
        rng=None,
        reproduction_rate=3,
        executor=None,
        force_simplify: bool,
        fitness_function: callable,
        **kwargs,
    ):
        if rng is None:
            rng = np.random.Generator(SFC64())

        parents = parent_selector(
            population, rng=rng, fitness_function=fitness_function
        )

        new_gen = np.empty(
            shape=(population.shape[-1] * reproduction_rate,), dtype=Individual
        )
        if len(parents) < 2:
            log.error("Not enough parents to perform crossover.")
            log.error(f"Parents selector: {parent_selector}")
            new_gen = np.full_like(new_gen, parents[0])

        if executor:
            futures = []
            for i in np.arange(population.shape[-1] * reproduction_rate, step=2):
                parent1, parent2 = rng.choice(parents, size=2, replace=False)

                futures.append(
                    executor.submit(
                        cls.cross_parents,
                        parent1,
                        parent2,
                        rng.integers(np.iinfo(np.uint32).max, dtype=np.uint32),
                        force_simplify,
                    )
                )
            i = 0
            for future in as_completed(futures):
                new_individual1, new_individual2 = future.result()
                new_gen[i] = new_individual1
                i += 1
                new_gen[i] = new_individual2
                i += 1
        else:
            for i in np.arange(population.shape[-1] * reproduction_rate, step=2):
                parent1, parent2 = rng.choice(parents, size=2, replace=False)

                new_individual1, new_individual2 = cls.cross_parents(
                    parent1, parent2, rng, force_simplify
                )

                new_gen[i] = new_individual1
                new_gen[i + 1] = new_individual2
        return new_gen

    @classmethod
    def cross_parents(
        cls, parent1, parent2, rng, force_simplify: bool = False
    ) -> tuple[Individual, Individual]:
        if isinstance(rng, np.integer):
            rng = np.random.Generator(SFC64(seed=rng))

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

        if force_simplify:
            new_individual1.simplify()
            new_individual2.simplify()
        return new_individual1, new_individual2


class Mutation(GeneticOperator):
    @classmethod
    def get_new_generation(
        cls,
        population,
        *,
        mutation_rate=1,
        rng=None,
        executor: Executor = None,
        force_simplify: bool,
        fitness_function: callable = None,
        **kwargs,
    ):
        if rng is None:
            rng = np.random.Generator(SFC64())
        elif isinstance(rng, np.integer):
            rng = np.random.Generator(SFC64(seed=rng))

        new_gen = np.empty(shape=(population.shape[-1],), dtype=Individual)
        if executor:
            futures = []
            for individual in population:
                if rng.random() < mutation_rate:
                    f = executor.submit(
                        cls.mutate,
                        individual.clone(),
                        rng.integers(np.iinfo(np.uint32).max, dtype=np.uint32),
                        force_simplify,
                        **kwargs,
                    )
                    futures.append(f)
            for i, future in enumerate(as_completed(futures)):
                new_gen[i] = future.result()

        else:
            for i, individual in enumerate(population):
                if rng.random() < mutation_rate:
                    ind = cls.mutate(
                        individual=individual.clone(),
                        rng=rng,
                        force_simplify=force_simplify,
                        **kwargs,
                    )
                    new_gen[i] = ind
        return new_gen

    @classmethod
    def mutate(cls, individual: Individual, rng, force_simplify=False, **kwargs):
        if isinstance(rng, np.integer):
            rng = np.random.Generator(SFC64(seed=rng))
        new_ind = cls._mutate(individual, rng, **kwargs)

        if force_simplify:
            new_ind.simplify()
        return new_ind

    @classmethod
    def _mutate(cls, individual, rng, **kwargs):
        raise NotImplementedError(f"{cls.__name__} is not yet implemented.")


class SubtreeMutation(Mutation):
    @classmethod
    def _mutate(cls, individual, rng, **kwargs):
        # Pick a random node to replace
        # Generate a new random subtree
        # Replace the node with the subtree

        node = rng.choice(individual.nodes)

        subtree = Individual(
            initialization_method="grow",
            max_depth=individual.depth - node.depth,
            input_size=individual.input_size,
        ).root
        if node.parent is None:
            individual.root = subtree
        else:
            node.parent._children.remove(node)
            node.parent.append(subtree)

        return individual


class PointMutation(Mutation):
    @classmethod
    def _mutate(cls, individual, rng, input_size: int, **kwargs):
        node = rng.choice(individual.nodes)
        new_type = rng.choice(reverse_valid_children[valid_children[node.type]])
        node.type = new_type
        if node.type == NodeType.CONSTANT:
            node.value = rng.random()
        elif node.type == NodeType.VARIABLE:
            node.value = rng.integers(input_size)

        return individual


class PermutationMutation(Mutation):
    @classmethod
    def _mutate(cls, individual, rng, **kwargs):
        valid_nodes = [
            node for node in individual.nodes if valid_children[node.type] == 2
        ]
        if valid_nodes:
            node = rng.choice(valid_nodes)
            while valid_children[node.type] != 2:
                node = rng.choice(valid_nodes)
            node.children[0], node.children[1] = node.children[1], node.children[0]

        return individual


class HoistMutation(Mutation):
    @classmethod
    def _mutate(cls, individual, rng, **kwargs):
        valid_nodes = [node for node in individual.nodes if node.parent is not None]
        if valid_nodes:
            node = rng.choice(valid_nodes)
            individual.root.type = node.type
            individual.root.value = node.value
            individual.root._children = node.children

            nodes = [
                individual.root,
            ]
            while nodes:
                node = nodes.pop()
                for child in node._children:
                    child.depth = node.depth + 1
                    nodes.append(child)

        return individual


class ShrinkMutation(Mutation):
    @classmethod
    def _mutate(cls, individual, rng, **kwargs):
        node1 = rng.choice(individual.nodes)
        node2 = rng.choice(node1.flatten)
        if node1.parent is None:
            individual.root = node2
        else:
            node1.parent._children.remove(node1)
            node1.parent.append(node2)

        return individual


class CollapseMutation(Mutation):
    @classmethod
    def _mutate(cls, individual, rng, **kwargs):
        node = rng.choice(individual.nodes)
        # n_type = rng.choice(terminal_set)
        # new_node = individual.get_random_node(n_type)
        new_node = Node(NodeType.CONSTANT, rng.random())
        if node.parent is None:
            individual.root = new_node
        else:
            node.parent._children.remove(node)
            node.parent.append(new_node)

        return individual


class ExpansionMutation(Mutation):
    @classmethod
    def _mutate(cls, individual, rng, **kwargs):
        valid_nodes = [
            node for node in individual.nodes if valid_children[node.type] == 0
        ]
        if valid_nodes:
            node = rng.choice(valid_nodes)
            subtree = Individual(
                initialization_method="grow",
                max_depth=individual.depth - node.depth,
                input_size=individual.input_size,
            ).root
            if node.parent is None:
                individual.root = subtree
            else:
                node.parent._children.remove(node)
                node.parent.append(subtree)

        return individual


class FineTuneMutation(Mutation):
    @classmethod
    def _mutate(cls, individual, rng, **kwargs):
        valid_nodes = [
            node for node in individual.nodes if node.type == NodeType.CONSTANT
        ]
        if valid_nodes:
            node = rng.choice(valid_nodes)
            node.value *= rng.normal(0.8, 1.2)

        return individual
