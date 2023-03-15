import numpy as np
from numpy.typing import NDArray
from numba import njit


@njit
def sort_population(
    genomes: NDArray, fitness: NDArray, valids: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    idx = np.argsort(fitness[:, 0])
    genomes = genomes[idx]
    fitness = fitness[idx]
    valids = valids[idx]
    return genomes, fitness, valids


@njit
def elitist(
    genomes: NDArray, fitness: NDArray, valids: NDArray, num_elites: int
) -> tuple[NDArray, NDArray, NDArray]:
    # Sort the population by fitness
    genomes_, fitness_, valids_ = sort_population(genomes, fitness, valids)
    # Return the top num_elites individuals
    return genomes_[:num_elites], fitness_[:num_elites], valids_[:num_elites]


def tournament():
    pass


def crowding_tournament():
    pass
