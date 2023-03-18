import pstats
import cProfile

import numpy as np

from problem import Problem, evaluate
from ga import GeneticAlgorithm, EpochConfig
from mutations import mutate_population
from population_manage import elitist, sort_population
from initializations import (
    generate_random_population,
    generate_cluster_genome,
    generate_cluster_population,
)
from utils import solution_to_list, solution_to_numpy

problem = Problem("data/train_0.json")
travel_times = problem.travel_times
capacity_nurse = problem.capacity_nurse
patients = problem.numpy_patients
coordinates = patients[:, :2]


def main():
    genome = generate_cluster_genome(
        n_nurses=25, n_patients=100, coordinates=coordinates
    )
    fitness, valid = evaluate(
        genome,
        travel_times,
        capacity_nurse,
        patients,
    )
    print(fitness)
    problem.visualize_solution(genome)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("profile/abc.prof")
