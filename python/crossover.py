from typing import Callable, Any

import numpy as np

import problem
from mutations import repair_greedy, repair_random


def crossover_greedy(
    genome1: np.ndarray, genome2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Destroy opposite paths and greedy insert lost patients."""
    # initiate children
    child1_ = np.copy(genome1)
    child2_ = np.copy(genome2)
    # choose nurse paths
    nurses = np.random.choice(genome1.shape[0], 2, replace=False)
    # destroy opposite paths
    child1_[nurses[0], :] = 0
    child2_[nurses[1], :] = 0
    # greedy insert lost patients
    child1 = repair_greedy(child1_, problem.travel_times)
    child2 = repair_greedy(child2_, problem.travel_times)

    return child1, child2


def crossover_random(
    genome1: np.ndarray, genome2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Destroy opposite paths and greedy insert lost patients."""
    # initiate children
    child1_ = np.copy(genome1)
    child2_ = np.copy(genome2)
    # choose nurse paths
    nurses = np.random.choice(genome1.shape[0], 2, replace=False)
    # destroy opposite paths
    child1_[nurses[0], :] = 0
    child2_[nurses[1], :] = 0
    # greedy insert lost patients
    child1 = repair_random(child1_)
    child2 = repair_random(child2_)

    return child1, child2


def deterministic_isolated_mating(
    population: np.ndarray, crossover_function: Callable[[Any], np.ndarray]
) -> np.ndarray:
    """Recieves a part of the population and mate every other genome"""
    pass


def stochastic_isolated_mating(
    population: np.ndarray, crossover_function: Callable[[Any], np.ndarray]
) -> np.ndarray:
    """Recieves a part of the population and mate everyone once randomly"""
    pass


def stochastic_elitist_mating(
    population: np.ndarray,
    fitness: np.ndarray,
    crossover_function: Callable[[Any], np.ndarray],
) -> np.ndarray:
    """Recieves a part of the population and mate randomly with emphasis on the best genomes. One genome can mate more than once."""
    pass


def mate():
    pass


def duel():
    pass


if __name__ == "__main__":
    from initializations import generate_random_genome
    from evaluations import evaluate

    genome1 = generate_random_genome(25, 100)
    genome2 = generate_random_genome(25, 100)
    child1, child2 = crossover_random(genome1, genome2)
    fitness1, _ = evaluate(child1)
    fitness2, _ = evaluate(child2)
    fit_parent1, _ = evaluate(genome1)
    fit_parent2, _ = evaluate(genome2)
    print(f"fitness child1: {fitness1}")
    print(f"fitness child2: {fitness2}")
    print(f"fitness parent1: {fit_parent1}")
    print(f"fitness parent2: {fit_parent2}")
