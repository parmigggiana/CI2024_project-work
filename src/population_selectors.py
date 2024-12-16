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


class FitnessProportionalSelector(Selector):
    @classmethod
    def select(cls, population, fitness_function, rng):
        p = np.array(
            [fitness_function(individual) for individual in population], dtype=float
        )
        p -= (
            np.min(p) - 1e-9
        )  # if all fitness are the same, the probability would be 0 without 1e-9
        p /= p.sum()
        p *= len(p)

        selected = population[rng.random(size=population.shape[-1]) < p]
        if selected.shape[-1] < 2:
            # If only one individual is selected, return the two highest-fitness individuals
            selected = population[np.argsort(p)[-2:]]
        return selected


class TournamentSelector(Selector):
    @classmethod
    def select(
        cls, population, tournaments, fitness_function, tournament_size, **kwargs
    ):
        selected = []
        for _ in range(tournaments):
            tournament = cls.rng.choice(population, size=tournament_size, replace=False)
            winner = max(tournament, key=lambda ind: fitness_function(ind))
            selected.append(winner)
        return np.array(selected)
