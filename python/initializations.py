import numpy as np
from numba import njit
from sklearn.cluster import KMeans

import problem


@njit
def generate_random_genome() -> np.ndarray:
    """Generate a random genome.

    Returns:
        np.ndarray: Numpy array filled with zeros and random patient ids up to limit.
    """
    patient_ids = np.arange(1, problem.nbr_patients + 1)
    genome = np.zeros((problem.nbr_nurses * problem.nbr_patients), dtype=np.int16)

    # Choose random indices to insert values from
    indices = np.random.choice(
        np.arange(problem.nbr_nurses * problem.nbr_patients),
        size=problem.nbr_patients,
        replace=False,
    )

    # Insert values from patient_ids into zeros
    genome[indices] = patient_ids

    # Reshape to (problem.nbr_nurses, problem.nbr_patients)
    genome = genome.reshape((problem.nbr_nurses, problem.nbr_patients))

    return genome


@njit
def generate_random_population(size: int) -> np.ndarray:
    """Generate a random population.

    Args:
        size (int): number of individuals in population

    Returns:
        np.ndarray: genomes
    """
    patient_ids = np.arange(1, problem.nbr_patients + 1)
    genome_ = np.zeros(
        (size * problem.nbr_nurses * problem.nbr_patients), dtype=np.int16
    )

    for idx in range(size):
        # Choose random indices to insert values from
        indices = (
            np.random.choice(
                np.arange(problem.nbr_nurses * problem.nbr_patients),
                size=problem.nbr_patients,
                replace=False,
            )
            + idx * problem.nbr_nurses * problem.nbr_patients
        )

        # Insert values from patient_ids into zeros
        genome_[indices] = patient_ids

    # Reshape
    genomes = genome_.reshape((size, problem.nbr_nurses, problem.nbr_patients))

    return genomes


def generate_cluster_genome() -> np.ndarray:
    """Generate a genome based on clustering.

    Returns:
        np.ndarray: genome
    """
    genome = np.zeros((problem.nbr_nurses, problem.nbr_patients), dtype=np.int16)
    # randomize how many clusters
    n_clusters = np.random.randint(problem.nbr_nurses // 1.35, problem.nbr_nurses + 1)
    # generate clusters with k nearest neighbors
    # settings for faster runtime
    neigh = KMeans(n_clusters=n_clusters, n_init=1, max_iter=25).fit(
        problem.coordinates
    )
    # get cluster labels
    cluster_labels = neigh.labels_
    # insert patients into genome
    for cluster_idx in range(n_clusters):
        # get patients in cluster
        patients_in_cluster = np.where(cluster_labels == cluster_idx)[0] + 1
        # shuffle patients
        np.random.shuffle(patients_in_cluster)
        # get patients to insert
        cluster_length = patients_in_cluster.shape[0]
        # insert patients
        genome[cluster_idx, :cluster_length] = patients_in_cluster
    return genome


def generate_cluster_population(size: int) -> np.ndarray:
    """Generate a population based on clustering.

    Args:
        size (int): number of individuals in population

    Returns:
        np.ndarray: genomes
    """
    # initialize population
    population = np.zeros(
        (size, problem.nbr_nurses, problem.nbr_patients), dtype=np.int16
    )
    # generate genomes
    for genome in population:
        genome_ = generate_cluster_genome()
        genome[:, :] = genome_[:, :]

    return population


# TODO fix
def generate_greedy_genome() -> np.ndarray:
    genome = np.zeros((problem.nbr_nurses, problem.nbr_patients), dtype=np.int16)
    first_patients = np.random.choice(
        np.arange(1, problem.nbr_patients + 1), size=problem.nbr_nurses, replace=False
    )
    for nurse_idx in range(problem.nbr_nurses):
        genome[nurse_idx, 0] = first_patients[nurse_idx]

    # fill in used patients
    used_patients = np.zeros(problem.nbr_patients, dtype=np.int16)
    used_patients[: first_patients.shape[0]] = first_patients
    used_patients_num = first_patients.shape[0]

    remaining_patients = np.setdiff1d(
        np.arange(1, problem.nbr_patients + 1), first_patients
    )

    # shuffle remaining patients
    np.random.shuffle(remaining_patients)
    for patient in remaining_patients:
        # find shortest travel times
        shortest_travel_time_idx = np.argsort(problem.travel_times)
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
            nurse_idx = np.random.randint(0, problem.nbr_nurses)
            spot = np.where(genome[nurse_idx] == 0)
            genome[nurse_idx, spot[0][0]] = patient
        used_patients[used_patients_num] = patient
        used_patients_num += 1
    return genome


# TODO fix
def generate_greedy_population(size: int) -> np.ndarray:
    """Generate a population based on greedy search."""
    population = np.zeros(
        (size, problem.nbr_nurses, problem.nbr_patients), dtype=np.int16
    )
    # generate each genome
    for genome in population:
        genome_ = generate_greedy_genome()
        genome[:, :] = genome_[:, :]

    return population


def generate_greedy_feasible_genome() -> np.ndarray:
    pass


if __name__ == "__main__":
    pass
    # import cProfile
    # import pstats

    # profiler = cProfile.Profile()
    # profiler.enable()

    # timing()

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats("profile/pop-gen.prof")
