import time

import numpy as np
from numpy.typing import NDArray

from solution_utils import solution_to_list, solution_to_numpy, generate_random_solution
from problem import Problem, evaluate

problem = Problem("data/train_0.json")
n_nurses = problem.nbr_nurses
n_patients = problem.nbr_patients


class Individual:
    def __init__(self) -> None:
        solution = generate_random_solution(n_nurses=n_nurses, n_patients=n_patients)
        self.dna = solution
        self.fitness = evaluate(
            solution=solution,
            travel_times=problem.travel_times,
            capacity_nurse=problem.capacity_nurse,
            patients=problem.numpy_patients,
            penalize_invalid=True,
        )


class GeneticAlgorithm:
    def __init__(self, size: int) -> None:
        self.population = self.create_population(size)
        self.size = size
        self.epoch = 0

    def create_population(
        self,
        size: int,
    ) -> list[Individual]:
        return [Individual() for _ in range(size)]

    def parent_selection(self, population: NDArray, num_parents: int) -> NDArray:
        # sort population
        population.sort(key=lambda x: x.fitness, reverse=True)
        # return the two first
        return population[:num_parents]

    def crossover(self):
        pass

    def mate(self):
        pass

    def tournament(population: list[Individual], pop_size: int) -> list[Individual]:
        population.sort(key=lambda x: x.fitness)

        # return survivors
        return population[:pop_size]

    def crowding_tournament(
        parents: list[Individual], children: list[Individual]
    ) -> list[Individual]:
        pass

    def duel(can1: Individual, can2: Individual) -> Individual:
        if can1.fitness > can2.fitness:
            return can2
        return can1

    def epoch(
        num_parents: int,
        num_children: int,
        deterministic: bool,
        ranked: bool,
        crowding: bool,
    ):
        average_fitnesses = []

        average_fitness = np.mean(np.array([x.fitness for x in self.population]))
        # display average fitness
        print(self.epoch, "average", average_fitness)
        # append to list for plotting
        average_fitnesses.append(average_fitness)
        # select best parents
        parents = parent_selection(
            population=population, num_parents=num_parents, maximize=maximize
        )

        if crowding:
            children = crossover(
                parents,
                bitstr_size=bitstr_size,
                num_children=num_children,
                deterministic=True,
                ranked=False,
                maximize=maximize,
            )

            survivors = crowding_tournament(parents, children, maximize)
        else:
            # perform crossover to get children
            children = crossover(
                parents,
                bitstr_size=bitstr_size,
                num_children=num_children,
                deterministic=deterministic,
                ranked=ranked,
                maximize=maximize,
            )

            # get survivors
            survivors = tournament(
                population=np.append(population, children).tolist(),
                pop_size=pop_size,
                maximize=maximize,
            )

        # update population
        population = survivors


if __name__ == "__main__":
    # CONFIG
    POPULATION_SIZE = 100
    NUM_EPISODES = 100

    MUTATE_MIN = 0
    MUTATE_MAX = 7
    GENOME_MUTATE_CHANCE = 0.4

    MAXIMIZE = True

    start = time.perf_counter()

    run(
        maximize=MAXIMIZE,
        pop_size=POPULATION_SIZE,
        num_ep=NUM_EPISODES,
        bitstr_size=BITSTRING_SIZE,
        # plot_ep=PLOT_EP,
        num_parents=50,
        num_children=50,
        deterministic=False,
        ranked=True,
        crowding=False,
    )

    end = time.perf_counter()
    print(f"time used {end-start}s")
