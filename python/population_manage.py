import numpy as np
from numpy.typing import NDArray


def elitist(
    genomes: NDArray, fitness: NDArray, valids: NDArray, num_elites: int
) -> tuple[NDArray, NDArray, NDArray]:
    # Sort the population by fitness
    idx = np.argsort(fitness[:, 0])
    genomes = genomes[idx]
    fitness = fitness[idx]
    valids = valids[idx]
    # Return the top num_elites individuals
    return genomes[:num_elites], fitness[:num_elites], valids[:num_elites]


def tournament():
    pass


def crowding_tournament():
    pass
