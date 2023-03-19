import numpy as np
from numba import njit


@njit
def sort_population(
    genomes: np.ndarray, fitness: np.ndarray, valids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.argsort(fitness[:, 0])
    genomes = genomes[idx]
    fitness = fitness[idx]
    valids = valids[idx]
    return genomes, fitness, valids


@njit
def elitist(
    genomes: np.ndarray, fitness: np.ndarray, valids: np.ndarray, num_elites: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Sort the population by fitness
    genomes_, fitness_, valids_ = sort_population(genomes, fitness, valids)
    # Return the top num_elites individuals
    return genomes_[:num_elites], fitness_[:num_elites], valids_[:num_elites]


# TODO finish
def parent_selection_feasible_mix(
    genomes: np.ndarray, fitness: np.ndarray, valids: np.ndarray, num_parents: int
) -> np.ndarray:
    # Sort the population by fitness
    genomes_, _, valids_ = sort_population(genomes, fitness, valids)
    # Return the parents
    return genomes_[:num_parents]


def tournament():
    pass


def crowding_tournament():
    pass
