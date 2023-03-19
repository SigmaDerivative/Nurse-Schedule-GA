from typing import Callable, Any

import numpy as np


def crossover_destroy_repair(
    genome1: np.ndarray,
    genome2: np.ndarray,
    n_destroys: int,
    repair_function: Callable[[Any], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Destroy opposite paths and repair lost patients."""
    # initiate children
    child1_ = np.copy(genome1)
    child2_ = np.copy(genome2)

    for _ in range(n_destroys):
        # choose nurse paths
        nurses = np.random.choice(genome1.shape[0], 2, replace=False)
        # destroy opposite paths
        child1_[nurses[0], :] = 0
        child2_[nurses[1], :] = 0
    # insert lost patients according to repair function
    child1 = repair_function(child1_)
    child2 = repair_function(child2_)

    return child1, child2


def deterministic_isolated_mating(
    population: np.ndarray,
    n_destroys: int,
    repair_function: Callable[[Any], np.ndarray],
) -> np.ndarray:
    """Recieves a population and mate every other genome"""
    if population.shape[0] % 2 != 0:
        raise ValueError("Population size must be even")
    children = np.zeros_like(population)
    for i in range(0, population.shape[0], 2):
        child1, child2 = crossover_destroy_repair(
            population[i], population[i + 1], n_destroys, repair_function
        )
        children[i] = child1
        children[i + 1] = child2

    return children


def stochastic_isolated_mating(
    population: np.ndarray,
    n_destroys: int,
    repair_function: Callable[[Any], np.ndarray],
) -> np.ndarray:
    """Recieves a part of the population and mate everyone once randomly"""
    pass


def stochastic_elitist_mating(
    population: np.ndarray,
    n_destroys: int,
    repair_function: Callable[[Any], np.ndarray],
    n_children: int,
    prob_factor: float = 3.0,
) -> np.ndarray:
    """Recieves a part of the population and mate randomly with emphasis on the best genomes. One genome can mate more than once."""
    if n_children % 2 != 0:
        raise ValueError("n_children must be even")
    children = np.zeros(
        (n_children, population.shape[1], population.shape[2]), dtype=np.int16
    )
    # calculate probabilities
    probabilities = (
        np.arange(population.shape[0], 0, -1) + population.shape[0] / prob_factor
    )
    probabilities = probabilities / np.sum(probabilities)

    for i in range(0, n_children, 2):
        # choose parents
        parents_idx = np.random.choice(
            population.shape[0], 2, replace=False, p=probabilities
        )
        child1, child2 = crossover_destroy_repair(
            population[parents_idx[0]],
            population[parents_idx[1]],
            n_destroys,
            repair_function,
        )
        children[i] = child1
        children[i + 1] = child2

    return children


def stochastic_mating(
    population: np.ndarray,
    n_destroys: int,
    repair_function: Callable[[Any], np.ndarray],
    n_children: int,
) -> np.ndarray:
    """Recieves a part of the population and mate randomly until n_children are created.
    One genome can mate more than once and same pair can also mate more than once."""
    if n_children % 2 != 0:
        raise ValueError("n_children must be even")
    children = np.zeros(
        (n_children, population.shape[1], population.shape[2]), dtype=np.int16
    )
    for i in range(0, n_children, 2):
        # choose parents
        parents_idx = np.random.choice(population.shape[0], 2, replace=False)
        child1, child2 = crossover_destroy_repair(
            population[parents_idx[0]],
            population[parents_idx[1]],
            n_destroys,
            repair_function,
        )
        children[i] = child1
        children[i + 1] = child2

    return children


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
