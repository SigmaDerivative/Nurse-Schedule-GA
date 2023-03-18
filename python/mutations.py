import numpy as np

from numba import njit
from tqdm import tqdm


# --- RANDOM INTRA MUTATIONS ---


# TODO: Fix
def intra_move_random(genome: np.ndarray) -> np.ndarray:
    """Randomly move one patient within a nurse path."""
    n_patients = genome.shape[1]
    n_nurses = genome.shape[0]
    patient_idx = np.random.randint(n_patients)
    nurse_idx = np.random.randint(n_nurses)
    shift = np.random.randint(3, n_patients)
    # shifted path
    genome[nurse_idx, :patient_idx] = genome[nurse_idx, :patient_idx]
    genome[nurse_idx, patient_idx + 1 :] = genome[nurse_idx, patient_idx:-1]
    genome[nurse_idx, patient_idx] = genome[nurse_idx, patient_idx + shift]
    genome[nurse_idx, patient_idx + shift] = genome[nurse_idx, patient_idx]

    return genome


@njit
def intra_move_swap(genome: np.ndarray) -> np.ndarray:
    """Swap two patients within a nurse path."""
    n_patients = genome.shape[1]
    n_nurses = genome.shape[0]
    chosen_nurse = np.random.randint(n_nurses)
    patient_idx1 = np.random.randint(n_patients)
    patient_idx2 = np.random.randint(n_patients)
    at_pos1 = genome[chosen_nurse, patient_idx1]
    at_pos2 = genome[chosen_nurse, patient_idx2]
    genome[chosen_nurse, patient_idx1] = at_pos2
    genome[chosen_nurse, patient_idx2] = at_pos1
    return genome


@njit
def roll_path(genome: np.ndarray) -> np.ndarray:
    """Randomly shift nurse path."""
    n_patients = genome.shape[1]
    n_nurses = genome.shape[0]
    chosen_nurse = np.random.randint(n_nurses)
    roll_amount = np.random.randint(n_patients)
    genome[chosen_nurse, :] = np.roll(genome[chosen_nurse, :], roll_amount)
    return genome


@njit
def shuffle_path(genome: np.ndarray) -> np.ndarray:
    """Randomly shuffle a nurse path."""
    n_nurses = genome.shape[0]
    chosen_nurse = np.random.randint(n_nurses)
    path = genome[chosen_nurse, :]
    np.random.shuffle(path)
    genome[chosen_nurse, :] = path
    return genome


@njit
def flip_path(genome: np.ndarray) -> np.ndarray:
    """Randomly flip a nurse path."""
    n_nurses = genome.shape[0]
    chosen_nurse = np.random.randint(n_nurses)
    path = genome[chosen_nurse, :]
    path_ = np.flip(path)
    genome[chosen_nurse, :] = path_
    return genome


# --- RANDOM INTER MUTATIONS ---


def number_interval_inter_swap(genome: np.ndarray) -> np.ndarray:
    """Swap numbers between patiens within their id interval."""
    pass


@njit
def index_interval_inter_swap(genome: np.ndarray) -> np.ndarray:
    """Swap patients between nurses based on indices in path."""
    # choose two nurse paths
    nurses = np.random.choice(genome.shape[0], 2, replace=False)
    # choose crossover point
    crossover = np.random.randint(1, genome.shape[1] - 1)
    # swap patients
    genome[nurses[0], :crossover], genome[nurses[1], :crossover] = (
        genome[nurses[1], :crossover],
        genome[nurses[0], :crossover],
    )
    genome[nurses[0], crossover:], genome[nurses[1], crossover:] = (
        genome[nurses[1], crossover:],
        genome[nurses[0], crossover:],
    )
    return genome


@njit
def random_swap(genome: np.ndarray) -> np.ndarray:
    """Swap patients (most likely) between nurses."""
    patients = np.random.choice(genome.shape[1], 2, replace=False)
    patient_1_idx_ = np.where(genome == patients[0])
    patient_1_idx = (patient_1_idx_[0][0], patient_1_idx_[1][0])
    patient_2_idx_ = np.where(genome == patients[1])
    patient_2_idx = (patient_2_idx_[0][0], patient_2_idx_[1][0])
    genome[patient_1_idx] = patients[1]
    genome[patient_2_idx] = patients[0]
    return genome


@njit
def random_multi_swap(genome: np.ndarray) -> np.ndarray:
    """Swap between 4 and 10 patients (most likely) between nurses."""
    num_swaps = np.random.randint(2, 6) * 2
    patients = np.random.choice(genome.shape[1], num_swaps, replace=False)
    for i in range(num_swaps // 2):
        patient_1_idx_ = np.where(genome == patients[i * 2])
        patient_1_idx = (patient_1_idx_[0][0], patient_1_idx_[1][0])
        patient_2_idx_ = np.where(genome == patients[i * 2 + 1])
        patient_2_idx = (patient_2_idx_[0][0], patient_2_idx_[1][0])
        genome[patient_1_idx] = patients[i * 2 + 1]
        genome[patient_2_idx] = patients[i * 2]
    return genome


# --- GREEDY MUTATIONS ---


@njit
def close_single_swap(
    genome: np.ndarray, travel_times: np.ndarray, m: int
) -> np.ndarray:
    """Swap a patient with a close patient. m is the number of closest patients to consider."""
    # choose patient to swap
    patient = np.random.randint(1, genome.shape[1] + 1)
    # find another close patient
    travel_times_ = travel_times[patient, :]
    travel_times_sorted = np.sort(travel_times_)
    # choose one of closest patients
    idx = np.random.randint(0, m)
    dist = travel_times_sorted[idx]
    target_patient = np.where(travel_times_ == dist)[0][0]

    # swap patients
    patient_1_idx_ = np.where(genome == patient)
    patient_1_idx = (patient_1_idx_[0][0], patient_1_idx_[1][0])
    patient_2_idx_ = np.where(genome == target_patient)
    patient_2_idx = (patient_2_idx_[0][0], patient_2_idx_[1][0])
    genome[patient_1_idx] = target_patient
    genome[patient_2_idx] = patient

    return genome


@njit
def far_single_swap(genome: np.ndarray, travel_times: np.ndarray, m: int) -> np.ndarray:
    """Swap a patient with a close patient. m is the number of furthest patients to consider."""
    # choose patient to swap
    patient = np.random.randint(1, genome.shape[1] + 1)
    # find another far patient
    travel_times_ = travel_times[patient, :]
    travel_times_sorted = np.sort(travel_times_)
    # choose one of furthest patients
    idx = np.random.randint(0, m)
    dist = travel_times_sorted[-idx]
    target_patient = np.where(travel_times_ == dist)[0][0]

    # swap patients
    patient_1_idx_ = np.where(genome == patient)
    patient_1_idx = (patient_1_idx_[0][0], patient_1_idx_[1][0])
    patient_2_idx_ = np.where(genome == target_patient)
    patient_2_idx = (patient_2_idx_[0][0], patient_2_idx_[1][0])
    genome[patient_1_idx] = target_patient
    genome[patient_2_idx] = patient

    return genome


# --- COMBINED MUTATIONS ---


@njit
def mutate_population(
    population: np.ndarray, travel_times: np.ndarray, m: int
) -> np.ndarray:
    """Mutate a population of genomes."""
    # genome_size = population.shape[1]
    for idx, genome in enumerate(population):
        # select mutation type
        mut_type = np.random.randint(0, 7)
        if mut_type == 0:
            genome_ = intra_move_swap(genome)
        elif mut_type == 1:
            genome_ = roll_path(genome)
        elif mut_type == 2:
            genome_ = shuffle_path(genome)
        elif mut_type == 3:
            genome_ = flip_path(genome)
        elif mut_type == 4:
            genome_ = random_multi_swap(genome)
        elif mut_type == 5:
            genome_ = close_single_swap(genome, travel_times=travel_times, m=m)
        elif mut_type == 6:
            genome_ = far_single_swap(genome, travel_times=travel_times, m=m)
        population[idx, :] = genome_
    return population


# --- TIMING ---
def timing():
    from problem import Problem

    problem = Problem("data/train_0.json")
    travel_times = problem.travel_times
    genome = generate_random_solution(n_patients=100, n_nurses=25)

    for _ in tqdm(range(100_000)):
        #     # genome = intra_move_random(genome)
        #     intra_move_swap(genome)
        #     roll_path(genome)
        #     shuffle_path(genome)
        #     flip_path(genome)
        #     index_interval_inter_swap(genome)
        genome = close_single_swap(genome, travel_times=travel_times, m=8)
        genome = far_single_swap(genome, travel_times=travel_times, m=8)


def timing2():
    population = generate_random_population(size=50, n_patients=100, n_nurses=25)

    for _ in tqdm(range(100_000)):
        # genome = intra_move_random(genome)
        population = mutate_population(population)


if __name__ == "__main__":
    from solution_utils import generate_random_solution
    import cProfile, pstats
    from solution_utils import generate_random_population

    profiler = cProfile.Profile()
    profiler.enable()

    timing()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("profile/mutations.prof")
