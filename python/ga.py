from dataclasses import dataclass

import numpy as np

import problem
from initializations import generate_random_population, generate_cluster_population
from evaluations import evaluate_population
from population_manage import elitist, sort_population
from mutations import mutate_population


@dataclass
class EpochConfig:
    num_survivors: int
    num_mutators: int


class GeneticAlgorithm:
    def __init__(self, size: int) -> None:
        genomes = generate_random_population(
            size=size, n_nurses=problem.nbr_nurses, n_patients=problem.nbr_patients
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
            size=n, n_nurses=problem.nbr_nurses, n_patients=problem.nbr_patients
        )
        new_fitness, new_valids = evaluate_population(new_genomes)

        # mutate part of survived genomes
        y = config.num_survivors - config.num_mutators
        mutated_genomes = mutate_population(population=surviver_genomes[y:], m=8)
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


def main():
    ga = GeneticAlgorithm(size=100)

    epoch_config = EpochConfig(num_survivors=95, num_mutators=94)

    for i in range(300):
        ga.epoch(epoch_config)

        if i % 10 == 0:
            ga.sort_population_()
            print(i, ga.fitness[0][0])

    problem.problem.visualize_solution(ga.genomes[0])


if __name__ == "__main__":
    import cProfile, pstats

    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("profile/export-data.prof")
