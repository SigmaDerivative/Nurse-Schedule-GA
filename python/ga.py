import time
import cProfile, pstats

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from numba import njit

from solution_utils import (
    generate_random_population,
    solution_to_list,
    solution_to_numpy,
)
from problem import Problem, evaluate
from population_manage import elitist, sort_population

problem = Problem("data/train_0.json")
nbr_nurses = problem.nbr_nurses
nbr_patients = problem.nbr_patients
travel_times = problem.travel_times
capacity_nurse = problem.capacity_nurse
numpy_patients = problem.numpy_patients


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
        num_survivors: int,
    ) -> None:
        # survival of the fittest
        surviver_genomes, surviver_fitness, surviver_valids = elitist(
            genomes=self.genomes,
            fitness=self.fitness,
            valids=self.valids,
            num_elites=num_survivors,
        )

        # create new individuals
        n = self.size - num_survivors
        new_genomes = generate_random_population(
            size=n, n_nurses=nbr_nurses, n_patients=nbr_patients
        )
        new_fitness, new_valids = evaluate_population(new_genomes)

        # update population
        genomes = np.vstack((surviver_genomes, new_genomes))
        fitness = np.vstack((surviver_fitness, new_fitness))
        valids = np.vstack((surviver_valids, new_valids))
        self.genomes = genomes
        self.fitness = fitness
        self.valids = valids

        # update epoch number
        self.epoch_number += 1


@njit
def njit_run(size: int, num_epochs: int, num_survivors: int) -> None:
    genomes = generate_random_population(
        size=size, n_nurses=nbr_nurses, n_patients=nbr_patients
    )
    fitness, valids = evaluate_population(genomes)

    for _ in range(num_epochs):
        # survival of the fittest
        surviver_genomes, surviver_fitness, surviver_valids = elitist(
            genomes=genomes,
            fitness=fitness,
            valids=valids,
            num_elites=num_survivors,
        )

        # create new individuals
        n = size - num_survivors
        new_genomes = generate_random_population(
            size=n, n_nurses=nbr_nurses, n_patients=nbr_patients
        )
        new_fitness, new_valids = evaluate_population(new_genomes)

        # update population
        genomes[:num_survivors] = surviver_genomes
        genomes[num_survivors:] = new_genomes
        fitness[:num_survivors] = surviver_fitness
        fitness[num_survivors:] = new_fitness
        valids[:num_survivors] = surviver_valids
        valids[num_survivors:] = new_valids


def main(pop_size: int):
    # ga = GeneticAlgorithm(size=pop_size)

    # for _ in tqdm(range(10_000)):
    #     ga.epoch(num_survivors=25)

    # ga.sort_population_()
    njit_run(size=pop_size, num_epochs=10_000, num_survivors=25)

    # print(ga.fitness)


def timing():
    times = []
    for i in range(100):
        start = time.perf_counter()

        ga = GeneticAlgorithm(size=POPULATION_SIZE)

        end = time.perf_counter()
        used_time = end - start
        times.append(used_time)
    print("mean time:", np.mean(times[1:]), "s")


if __name__ == "__main__":
    # CONFIG
    POPULATION_SIZE = 100

    profiler = cProfile.Profile()
    profiler.enable()

    before = time.perf_counter()

    main(pop_size=POPULATION_SIZE)

    after = time.perf_counter()
    print("time:", after - before, "s")

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.strip_dirs()
    # stats.print_stats()
    stats.dump_stats("profile/export-data.prof")
