import itertools

import numpy as np
from numpy.typing import NDArray
from numba import njit
from sklearn.cluster import KMeans


def solution_to_numpy(solution: list) -> NDArray:
    """Converts a solution to a numpy array.

    Code from: https://stackoverflow.com/questions/38619143/convert-python-sequence-to-numpy-array-filling-missing-values

    Args:
        solution (list): A potential solution.
        Solution is on format one row per nurse,
        with id for each patient visisted.

    Returns:
        NDArray: A numpy array, where each row is a nurse route.
    """
    # convert to numpy array
    solution_numpy = np.array(list(itertools.zip_longest(*solution, fillvalue=0))).T
    # pad with zeros
    num_patients = np.max(solution_numpy)
    pad_size = num_patients - solution_numpy.shape[1]
    solution_numpy = np.pad(
        solution_numpy, [(0, 0), (0, pad_size)], mode="constant", constant_values=0
    )

    # convert to int
    solution_numpy = solution_numpy.astype(np.int32)

    return solution_numpy


def solution_to_list(solution: NDArray) -> list:
    """Converts a solution to a list of lists.

    Args:
        solution (NDArray): A potential solution.
        Solution is on format one row per nurse,
        with id for each patient visisted,
        filled with zeros inbetween.

    Returns:
        list: A list of lists, where each list is a nurse route.
    """
    solution_list = []
    for nurse_path in solution:
        # remove zeros
        nurse_path = nurse_path[nurse_path != 0]
        # convert to list
        nurse_path = nurse_path.tolist()
        solution_list.append(nurse_path)
    return solution_list


@njit
def generate_random_solution(n_nurses: int, n_patients: int) -> NDArray:
    """Generate a random solution.

    Args:
        n_nurses (int): Number of rows
        n_patients (int): Number of columns

    Returns:
        NDArray: Numpy array filled with zeros and random patient ids up to limit.
    """
    patient_ids = np.arange(1, n_patients + 1)
    solution = np.zeros((n_nurses * n_patients), dtype=np.int16)

    # Choose random indices to insert values from
    indices = np.random.choice(
        np.arange(n_nurses * n_patients), size=n_patients, replace=False
    )

    # Insert values from patient_ids into zeros
    solution[indices] = patient_ids

    # Reshape to (n_nurses, n_patients)
    solution = solution.reshape((n_nurses, n_patients))

    return solution


@njit
def generate_random_population(size: int, n_nurses: int, n_patients: int) -> NDArray:
    """Generate a random population.

    Args:
        size (int): number of individuals in population
        n_nurses (int): determine size of genome
        n_patients (int): determine size of genome

    Returns:
        NDArray: genomes
    """
    patient_ids = np.arange(1, n_patients + 1)
    genome_ = np.zeros((size * n_nurses * n_patients), dtype=np.int16)

    for idx in range(size):
        # Choose random indices to insert values from
        indices = (
            np.random.choice(
                np.arange(n_nurses * n_patients), size=n_patients, replace=False
            )
            + idx * n_nurses * n_patients
        )

        # Insert values from patient_ids into zeros
        genome_[indices] = patient_ids

    # Reshape
    genomes = genome_.reshape((size, n_nurses, n_patients))

    return genomes


def generate_cluster_genome(
    n_nurses: int, n_patients: int, coordinates: np.ndarray
) -> np.ndarray:
    """Generate a genome based on clustering.

    Args:
        n_nurses (int): determine size of genome
        n_patients (int): determine size of genome
        coordinates (np.ndarray): coordinates of patients

    Returns:
        np.ndarray: genome
    """
    genome = np.zeros((n_nurses, n_patients), dtype=np.int16)
    # randomize how many clusters
    n_clusters = np.random.randint(n_nurses // 1.5, n_nurses + 1)
    # generate clusters with k nearest neighbors
    neigh = KMeans(n_clusters=n_clusters, random_state=0).fit(coordinates)
    # get cluster labels
    cluster_labels = neigh.labels_
    # insert patients into genome
    for nurse_idx in range(n_nurses):
        # get patients in cluster
        patients_in_cluster = np.where(cluster_labels == nurse_idx)[0]
        # shuffle patients
        np.random.shuffle(patients_in_cluster)
        # get patients to insert
        cluster_length = patients_in_cluster.shape[0]
        # insert patients
        genome[nurse_idx, :cluster_length] = patients_in_cluster
    return genome


def generate_cluster_population(
    size: int, n_nurses: int, n_patients: int, travel_times: np.ndarray
) -> NDArray:
    """Generate a population based on clustering.

    Args:
        size (int): number of individuals in population
        n_nurses (int): determine size of genome
        n_patients (int): determine size of genome
        travel_times (np.ndarray): travel times between patients

    Returns:
        NDArray: genomes
    """
    pass


# TODO fix
def generate_greedy_genome(
    n_nurses: int, n_patients: int, travel_times: np.ndarray
) -> np.ndarray:
    genome = np.zeros((n_nurses, n_patients), dtype=np.int16)
    first_patients = np.random.choice(
        np.arange(1, n_patients + 1), size=n_nurses, replace=False
    )
    for nurse_idx in range(n_nurses):
        genome[nurse_idx, 0] = first_patients[nurse_idx]

    # fill in used patients
    used_patients = np.zeros(n_patients, dtype=np.int16)
    used_patients[: first_patients.shape[0]] = first_patients
    used_patients_num = first_patients.shape[0]

    remaining_patients = np.setdiff1d(np.arange(1, n_patients + 1), first_patients)

    # shuffle remaining patients
    np.random.shuffle(remaining_patients)
    for patient in remaining_patients:
        # find shortest travel times
        shortest_travel_time_idx = np.argsort(travel_times)
        # pick one of the shortest travel times
        idx_pick = np.random.randint(0, 5)
        shortest_travel_time_idx = shortest_travel_time_idx[idx_pick]
        if shortest_travel_time_idx in used_patients:
            # insert as next in path (get spot to put after)
            spot = np.where(genome == shortest_travel_time_idx)
            # make space
            genome[spot[0][0], spot[0][1] + 2 : spot[0][1] + 11] = genome[
                spot[0][0], spot[0][1] + 1 : spot[0][1] + 10
            ]
            # insert
            genome[spot[0][0], spot[0][1] + 1] = patient
        else:
            # insert randomly
            nurse_idx = np.random.randint(0, n_nurses)
            spot = np.where(genome[nurse_idx] == 0)
            genome[nurse_idx, spot[0][0]] = patient
        used_patients[used_patients_num] = patient
        used_patients_num += 1
    return genome


# TODO fix
def generate_greedy_population(
    size: int, n_nurses: int, n_patients: int, travel_times: np.ndarray
) -> NDArray:
    """Generate a population based on greedy search."""
    population = np.zeros((size, n_nurses, n_patients), dtype=np.int16)
    # assign one random patient to each nurse
    for genome in population:
        genome_ = generate_greedy_genome(
            n_nurses=n_nurses, n_patients=n_patients, travel_times=travel_times
        )
        genome[:, :] = genome_[:, :]

    # assign remaining patients to nurse with shortest travel time
    return population


def timing():
    from problem import Problem

    problem = Problem("data/train_0.json")
    travel_times = problem.travel_times

    pop1 = generate_random_population(size=100, n_nurses=25, n_patients=100)
    pop2 = generate_cluster_population(
        size=100, n_nurses=25, n_patients=100, travel_times=travel_times
    )


def testing():
    from problem import Problem
    from ga import evaluate_population, evaluate
    from population_manage import sort_population

    problem = Problem("data/train_0.json")
    travel_times = problem.travel_times
    capacity_nurse = problem.capacity_nurse
    patients = problem.numpy_patients
    coordinates = patients[:, :2]

    genome = generate_cluster_genome(
        n_nurses=25, n_patients=100, coordinates=coordinates
    )

    # population = generate_greedy_population(
    #     size=100, n_nurses=25, n_patients=100, travel_times=travel_times
    # )
    # fitness, valids = evaluate_population(population)
    # population, fitness, valids = sort_population(
    #     genomes=population,
    #     fitness=fitness,
    #     valids=valids,
    # )
    fitness, valid = evaluate(
        genome,
        travel_times,
        capacity_nurse,
        patients,
    )
    print(fitness)
    problem.visualize_solution(genome)


if __name__ == "__main__":
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    testing()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("profile/pop-gen.prof")
