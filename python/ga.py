from typing import Callable, Any
from dataclasses import dataclass

import numpy as np

import problem
from initializations import generate_random_population, generate_cluster_population
from evaluations import evaluate_population
from population_manage import elitist, sort_population
from mutations import mutate_population, repair_random, repair_greedy
from crossover import deterministic_isolated_mating, stochastic_elitist_mating


@dataclass
class EpochConfig:
    num_parents: int
    num_new_clustered_individuals: int
    num_new_random_individuals: int
    n_destroys: int
    repair_function: Callable[[Any], np.ndarray]
    n_children: int
    mate_elite_prob_factor: float = 2.0
    mutation_m: int = 4


class GeneticAlgorithm:
    def __init__(self, size: int) -> None:
        genomes = generate_random_population(size=size)
        fitness, valids = evaluate_population(genomes)
        self.genomes = genomes
        self.fitness = fitness
        self.valids = valids
        self.size = size
        self.epoch_number = 0

    def __add__(self, other: "GeneticAlgorithm") -> "GeneticAlgorithm":
        genomes = np.concatenate((self.genomes, other.genomes))
        fitness = np.concatenate((self.fitness, other.fitness))
        valids = np.concatenate((self.valids, other.valids))
        ga = GeneticAlgorithm(size=0)
        ga.genomes = genomes
        ga.fitness = fitness
        ga.valids = valids
        ga.size = self.size + other.size
        ga.epoch_number = self.epoch_number + other.epoch_number
        return ga

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
        # get parent candidates
        parent_genomes, _, _ = elitist(
            genomes=self.genomes,
            fitness=self.fitness,
            valids=self.valids,
            num_elites=config.num_parents,
        )

        # random repair function
        if np.random.uniform() < 0.5:
            repair_func = repair_random
        else:
            repair_func = repair_greedy

        # crossover
        if np.random.uniform() < 0.5:
            child_genomes = deterministic_isolated_mating(
                parent_genomes, config.n_destroys, repair_func
            )
        else:
            child_genomes = stochastic_elitist_mating(
                parent_genomes,
                config.n_destroys,
                repair_func,
                config.n_children,
                config.mate_elite_prob_factor,
            )
        mutated_genomes = mutate_population(population=child_genomes, m=8)

        # create new individuals
        if config.num_new_clustered_individuals > 0:
            new_genomes_cluster = generate_cluster_population(
                size=config.num_new_clustered_individuals
            )
            new_genomes_cluster = mutate_population(
                population=new_genomes_cluster, m=config.mutation_m
            )
        new_genomes_random = generate_random_population(
            size=config.num_new_random_individuals
        )
        new_genomes_random = mutate_population(
            population=new_genomes_random, m=config.mutation_m
        )

        # combine all genomes
        if config.num_new_clustered_individuals > 0:
            total_genomes = np.vstack(
                (
                    parent_genomes,
                    new_genomes_cluster,
                    new_genomes_random,
                    mutated_genomes,
                )
            )
        else:
            total_genomes = np.vstack(
                (parent_genomes, new_genomes_random, mutated_genomes)
            )
        total_fitness, total_valids = evaluate_population(total_genomes)
        # survival of the fittest
        surviver_genomes, surviver_fitness, surviver_valids = elitist(
            genomes=total_genomes,
            fitness=total_fitness,
            valids=total_valids,
            num_elites=self.size,
        )

        # update population
        self.genomes = surviver_genomes
        self.fitness = surviver_fitness
        self.valids = surviver_valids

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


@dataclass
class TrainConfig:
    pop_size: int
    num_epoch: int
    print_num: int


def train_population(
    train_config: TrainConfig, epoch_config: EpochConfig
) -> GeneticAlgorithm:
    ga = GeneticAlgorithm(size=train_config.pop_size)

    for idx in range(train_config.num_epoch):
        ga.epoch(epoch_config)
        if idx % train_config.print_num == 0:
            ga.sort_population_()
            print(f"epoch {idx} fitness {ga.fitness[0][0]}")

    return ga


def main():
    epoch_config = EpochConfig(
        num_parents=20,
        num_new_clustered_individuals=2,
        num_new_random_individuals=4,
        n_destroys=2,
        repair_function=repair_random,
        n_children=20,
    )
    train_config = TrainConfig(pop_size=100, num_epoch=1000, print_num=10)

    ga = train_population(train_config=train_config, epoch_config=epoch_config)

    ga.sort_population_()
    print(f"fitness {ga.fitness[0][0]}")
    problem.problem.visualize_solution(ga.genomes[0])


if __name__ == "__main__":
    import cProfile, pstats

    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("profile/export-data.prof")
