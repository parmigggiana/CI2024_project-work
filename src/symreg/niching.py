import logging
from abc import ABC

log = logging.getLogger(__name__)


class NichingStrategy(ABC):
    @classmethod
    def run(cls, population, **kwargs):
        raise NotImplementedError(f"{cls.__name__} is not yet implemented.")


class Extinction(NichingStrategy):
    @classmethod
    def run(cls, gp, **kwargs):
        unique = set(gp.population)
        if len(unique) <= 2:
            log.info("Extinction event occurred. Reinitializing population.")
            for i, ind in enumerate(unique, 1):
                gp.population[-i] = ind
            gp.init_population(
                gp.population_size - len(unique), gp.init_max_depth, gp.executor
            )


class Island(NichingStrategy): ...


class HierarchicalFairCompetition(NichingStrategy): ...


class DeterministicCrowding(NichingStrategy): ...


class AllopatricSelection(NichingStrategy): ...


class StandardCrowding(NichingStrategy): ...


class RestrictedTournamentSelection(NichingStrategy): ...


class Gender(NichingStrategy): ...


class RandomImmigrants(NichingStrategy): ...


class TwoLevelDiversitySelection(NichingStrategy): ...
