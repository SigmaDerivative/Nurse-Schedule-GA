import time

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from solution_utils import solution_to_list, solution_to_numpy, generate_random_solution
from problem import Problem, evaluate
from population_manage import elitist

problem = Problem("data/train_0.json")


def generate_random_population(size: int) -> tuple[NDArray, NDArray, NDArray]:
    """Generate a random population.

    Args:
        n_nurses (int): determine size of genome
        NDArray (int): determine size of genome
        size (int): number of individuals in population

    Returns:
        tuple[NDArray, NDArray]: genomes, fitness, valids
    """
    genome = generate_random_solution(
        n_nurses=problem.nbr_nurses, n_patients=problem.nbr_patients
    )
    fitness, valid = evaluate(
        genome,
        travel_times=problem.travel_times,
        capacity_nurse=problem.capacity_nurse,
        patients=problem.numpy_patients,
        penalize_invalid=True,
    )
    genome = genome[np.newaxis, :]

    for _ in range(1, size):
        genome_ = generate_random_solution(
            n_nurses=problem.nbr_nurses, n_patients=problem.nbr_patients
        )
        fitness_, valid_ = evaluate(
            genome_,
            problem.travel_times,
            problem.capacity_nurse,
            problem.numpy_patients,
            penalize_invalid=True,
        )
        genome_ = genome_[np.newaxis, :]
        genome = np.vstack((genome, genome_))
        fitness = np.vstack((fitness, fitness_))
        valid = np.vstack((valid, valid_))

    return genome, fitness, valid


class GeneticAlgorithm:
    def __init__(self, size: int) -> None:
        genomes, fitness, valids = generate_random_population(size=size)
        self.genomes = genomes
        self.fitness = fitness
        self.valids = valids
        self.size = size
        self.epoch_number = 0

    def sort_population_(self) -> None:
        # sort population
        idx = np.argsort(self.fitness[:, 0])
        self.genomes = self.genomes[idx]
        self.fitness = self.fitness[idx]
        self.valids = self.valids[idx]

    def epoch(
        self,
        num_survivors: int,
    ) -> None:
        # survival of the fittest
        genomes, fitness, valids = elitist(
            genomes=self.genomes,
            fitness=self.fitness,
            valids=self.valids,
            num_elites=num_survivors,
        )

        # create new individuals
        n = self.size - num_survivors
        new_genomes, new_fitness, new_valids = generate_random_population(size=n)

        # update population
        genomes = np.vstack((genomes, new_genomes))
        fitness = np.vstack((fitness, new_fitness))
        valids = np.vstack((valids, new_valids))

        self.genomes = genomes
        self.fitness = fitness
        self.valids = valids

        # update epoch number
        self.epoch_number += 1


def main(pop_size: int):
    ga = GeneticAlgorithm(size=pop_size)

    for _ in tqdm(range(10_000)):
        ga.epoch(num_survivors=25)

    ga.sort_population_()

    print(ga.genomes.shape, ga.fitness.shape, ga.valids.shape)


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

    main(pop_size=POPULATION_SIZE)
