import numpy as np
from numba import njit
from sklearn.cluster import KMeans
from tqdm import tqdm


@njit
def generate_random_genome(n_nurses: int, n_patients: int) -> np.ndarray:
    """Generate a random genome.

    Args:
        n_nurses (int): Number of rows
        n_patients (int): Number of columns

    Returns:
        np.ndarray: Numpy array filled with zeros and random patient ids up to limit.
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
def generate_random_population(size: int, n_nurses: int, n_patients: int) -> np.ndarray:
    """Generate a random population.

    Args:
        size (int): number of individuals in population
        n_nurses (int): determine size of genome
        n_patients (int): determine size of genome

    Returns:
        np.ndarray: genomes
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
    n_clusters = np.random.randint(n_nurses // 1.3, n_nurses + 1)
    # generate clusters with k nearest neighbors
    # settings for faster runtime
    neigh = KMeans(n_clusters=n_clusters, n_init=1, max_iter=25).fit(coordinates)
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


def generate_cluster_population(
    size: int, n_nurses: int, n_patients: int, coordinates: np.ndarray
) -> np.ndarray:
    """Generate a population based on clustering.

    Args:
        size (int): number of individuals in population
        n_nurses (int): determine size of genome
        n_patients (int): determine size of genome
        coordinates (np.ndarray): coordinates of patients

    Returns:
        np.ndarray: genomes
    """
    # initialize population
    population = np.zeros((size, n_nurses, n_patients), dtype=np.int16)
    # generate genomes
    for genome in population:
        genome_ = generate_cluster_genome(
            n_nurses=n_nurses, n_patients=n_patients, coordinates=coordinates
        )
        genome[:, :] = genome_[:, :]

    return population


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


def generate_greedy_feasible_genome(
    patients: np.ndarray, trael_times: np.ndarray, nurse_capacity: int
) -> np.ndarray:
    pass


# TODO fix
def generate_greedy_population(
    size: int, n_nurses: int, n_patients: int, travel_times: np.ndarray
) -> np.ndarray:
    """Generate a population based on greedy search."""
    population = np.zeros((size, n_nurses, n_patients), dtype=np.int16)
    # generate each genome
    for genome in population:
        genome_ = generate_greedy_genome(
            n_nurses=n_nurses, n_patients=n_patients, travel_times=travel_times
        )
        genome[:, :] = genome_[:, :]

    return population


def timing():
    from problem import Problem, evaluate
    from ga import evaluate_population

    problem = Problem("data/train_0.json")
    travel_times = problem.travel_times
    capacity_nurse = problem.capacity_nurse
    patients = problem.numpy_patients
    coordinates = problem.numpy_patients[:, :2]

    # for _ in tqdm(range(10)):
    #     pop = generate_cluster_population(
    #         size=100, n_nurses=25, n_patients=100, coordinates=coordinates
    #     )
    genome = generate_cluster_genome(
        n_nurses=25, n_patients=100, coordinates=coordinates
    )
    # pop = generate_cluster_population(
    #     size=3, n_nurses=25, n_patients=100, coordinates=coordinates
    # )
    # fitnesses, _ = evaluate_population(pop)
    # print(fitnesses[0])
    fit, _ = evaluate(genome, travel_times, capacity_nurse, patients)
    print(fit)
    problem.visualize_solution(genome)
    # problem.visualize_solution(pop[0])


if __name__ == "__main__":
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    timing()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("profile/pop-gen.prof")
