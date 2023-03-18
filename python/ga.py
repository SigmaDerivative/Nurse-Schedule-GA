import time
import cProfile, pstats
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from numba import njit

from initializations import generate_random_population, generate_cluster_population
from problem import Problem, evaluate
from population_manage import elitist, sort_population
from mutations import mutate_population
from utils import solution_to_list, solution_to_numpy

problem = Problem("data/train_0.json")
nbr_nurses = problem.nbr_nurses
nbr_patients = problem.nbr_patients
travel_times = problem.travel_times
capacity_nurse = problem.capacity_nurse
numpy_patients = problem.numpy_patients
coordinates = numpy_patients[:, :2]


@dataclass
class EpochConfig:
    num_survivors: int
    num_mutators: int


@njit
def evaluate_population(genomes: NDArray) -> NDArray:
    fitness = np.empty((genomes.shape[0], 1), dtype=np.float64)
    valid = np.empty((genomes.shape[0], 1), dtype=np.bool_)
    for idx, genome in enumerate(genomes):
        fitness_, valid_ = evaluate(
            genome, travel_times, capacity_nurse, numpy_patients, penalize_invalid=True
        )
        fitness[idx, 0] = fitness_
        valid[idx, 0] = valid_
    return fitness, valid


class GeneticAlgorithm:
    def __init__(self, size: int) -> None:
        genomes = generate_random_population(
            size=size, n_nurses=nbr_nurses, n_patients=nbr_patients
        )
        fitness, valids = evaluate_population(genomes)
        self.genomes = genomes
        self.fitness = fitness
        self.valids = valids
        self.size = size
        self.epoch_number = 0

    def sort_population_(self) -> None:
        genomes, fitness, valids = sort_population(
            genomes=self.genomes,
            fitness=self.fitness,
            valids=self.valids,
        )
        self.genomes = genomes
        self.fitness = fitness
        self.valids = valids

    def epoch(
        self,
        config: EpochConfig,
    ) -> None:
        # survival of the fittest
        surviver_genomes, surviver_fitness, surviver_valids = elitist(
            genomes=self.genomes,
            fitness=self.fitness,
            valids=self.valids,
            num_elites=config.num_survivors,
        )

        # create new individuals
        n = self.size - config.num_survivors
        new_genomes = generate_cluster_population(
            size=n,
            n_nurses=nbr_nurses,
            n_patients=nbr_patients,
            coordinates=coordinates,
        )
        new_fitness, new_valids = evaluate_population(new_genomes)

        # mutate part of survived genomes
        y = config.num_survivors - config.num_mutators
        mutated_genomes = mutate_population(
            population=surviver_genomes[y:],
            travel_times=travel_times,
            m=8,
        )
        mutated_fitness, mutated_valids = evaluate_population(mutated_genomes)

        # update population
        self.genomes = np.vstack((surviver_genomes[:y], mutated_genomes, new_genomes))
        self.fitness = np.vstack((surviver_fitness[:y], mutated_fitness, new_fitness))
        self.valids = np.vstack((surviver_valids[:y], mutated_valids, new_valids))

        # update epoch number
        self.epoch_number += 1


# @njit
# def njit_run(size: int, num_epochs: int, num_survivors: int) -> None:
#     genomes = generate_random_population(
#         size=size, n_nurses=nbr_nurses, n_patients=nbr_patients
#     )
#     fitness, valids = evaluate_population(genomes)

#     for _ in range(num_epochs):
#         # survival of the fittest
#         surviver_genomes, surviver_fitness, surviver_valids = elitist(
#             genomes=genomes,
#             fitness=fitness,
#             valids=valids,
#             num_elites=num_survivors,
#         )

#         # create new individuals
#         n = size - num_survivors
#         new_genomes = generate_random_population(
#             size=n, n_nurses=nbr_nurses, n_patients=nbr_patients
#         )
#         new_fitness, new_valids = evaluate_population(new_genomes)

#         # update population
#         genomes[:num_survivors] = surviver_genomes
#         genomes[num_survivors:] = new_genomes
#         fitness[:num_survivors] = surviver_fitness
#         fitness[num_survivors:] = new_fitness
#         valids[:num_survivors] = surviver_valids
#         valids[num_survivors:] = new_valids


# def main_njit(pop_size: int):
#     njit_run(size=pop_size, num_epochs=10_000, num_survivors=25)


def main(pop_size: int):
    ga = GeneticAlgorithm(size=pop_size)

    epoch_config = EpochConfig(num_survivors=95, num_mutators=94)

    for i in range(300):
        ga.epoch(epoch_config)

        if i % 10 == 0:
            ga.sort_population_()
            print(i, ga.fitness[0][0])

    problem.visualize_solution(ga.genomes[0])


if __name__ == "__main__":
    # CONFIG
    POPULATION_SIZE = 100

    profiler = cProfile.Profile()
    profiler.enable()

    main(pop_size=POPULATION_SIZE)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("profile/export-data.prof")
