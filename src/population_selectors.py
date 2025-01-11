import itertools
from random import choice
import warnings

import numpy as np


class Selector:
    @classmethod
    def select(cls, population: np.array, **kwargs) -> np.array:
        raise NotImplementedError(f"{cls.__name__} is not yet implemented.")


class DeterministicSelector(Selector):
    @classmethod
    def select(cls, population, size, fitness_function, **kwargs):
        return np.array(
            sorted(population, key=lambda ind: fitness_function(ind), reverse=True)[
                :size
            ]
        )


class BalancedDeterministicSelector(Selector):
    @classmethod
    def select(cls, population, size, fitness_function, **kwargs):
        if len(population) < 800:
            return DeterministicSelector.select(population, size, fitness_function)

        sorted_population = sorted(
            population, key=lambda ind: fitness_function(ind), reverse=True
        )
        # Select the best 80% of size and 20% from the worst third
        split = int(size * 0.8)
        best = sorted_population[:split]
        split = size - split
        worst = sorted_population[2 * size // 3 : 2 * size // 3 + split]
        return np.array(best + worst)


class FitnessHoleSelector(Selector):
    @classmethod
    def select(cls, population, size, fitness_function, rng, **kwargs):
        # Instead of tournament with 2 individuals
        # use n individuals but with fixed amount of tournaments so the population size is constant
        selected = []
        for tournament in np.array_split(population, size):
            if len(tournament) == 1:
                selected.append(tournament[0])
                continue

            fittest = np.argmax([fitness_function(ind) for ind in tournament])
            shallowest = np.argmin([ind.depth for ind in tournament])
            if rng.random() < 0.8:
                ind = tournament[fittest]
            else:
                ind = tournament[shallowest]
            selected.append(ind)

        return np.array(selected)


class FitnessProportionalSelector(Selector):
    @classmethod
    def select(cls, population, fitness_function, rng):
        p = np.array(
            [fitness_function(individual) for individual in population],
            dtype=float,
        )
        p -= np.min(p) - 1e-9
        p = np.maximum(p, 0)
        p /= p.sum()

        p *= len(p)

        selected = population[rng.random(size=population.shape[-1]) < p]
        if selected.shape[-1] < 2:
            # If only one individual is selected, return the two highest-fitness individuals
            selected = population[np.argsort(p)[-2:]]
        return selected


class TournamentSelector(Selector):
    @classmethod
    def select(cls, population, fitness_function, tournaments=None, **kwargs):
        if tournaments is None:
            tournaments = int(np.sqrt(len(population)))

        selected = []
        # Split population into tournaments
        # use numpy array split to split the population into tournaments
        # then use np.argmax to find the index of the winner of each tournament
        # then use that index to select the winner from the population
        for tournament in np.array_split(population, tournaments):
            winner = tournament[
                np.argmax([fitness_function(ind) for ind in tournament])
            ]
            selected.append(winner)

        return np.array(selected)
