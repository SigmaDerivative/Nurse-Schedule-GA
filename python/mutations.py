import numpy as np

from numba import njit

import problem
from evaluations import evaluate


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
def close_single_swap(genome: np.ndarray, m: int) -> np.ndarray:
    """Swap a patient with a close patient. m is the number of closest patients to consider."""
    # choose patient to swap
    patient = np.random.randint(1, genome.shape[1] + 1)
    # find another close patient
    travel_times_ = problem.travel_times[patient, :]
    travel_times_sorted = np.sort(travel_times_)
    # choose one of closest patients
    idx = np.random.randint(1, m)
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
def far_single_swap(genome: np.ndarray, m: int) -> np.ndarray:
    """Swap a patient with a close patient. m is the number of furthest patients to consider."""
    # choose patient to swap
    patient = np.random.randint(1, genome.shape[1] + 1)
    # find another far patient
    travel_times_ = problem.travel_times[patient, :]
    travel_times_sorted = np.sort(travel_times_)
    # choose one of furthest patients
    idx = np.random.randint(1, m)
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


def close_multi_swap(genome: np.ndarray, m: int, max: int) -> np.ndarray:
    """Swap a patient with a close patient.
    m is the number of closest patients to consider.
    max is the maximum number of patients to swap."""
    pass


def far_multi_swap(genome: np.ndarray, m: int, max: int) -> np.ndarray:
    """Swap a patient with a close patient.
    m is the number of furthest patients to consider.
    max is the maximum number of patients to swap."""
    pass


def time_close_single_swap(genome: np.ndarray, m: int) -> np.ndarray:
    """Swap a patient with another patient with close start time. m is the number of patients to consider."""
    pass


def time_far_single_swap(genome: np.ndarray, m: int) -> np.ndarray:
    """Swap a patient with another patient with far off start time. m is the number of patients to consider."""
    pass


def expediate_early_ends(genome: np.ndarray) -> np.ndarray:
    """Move patients with early end times to the front of the schedule."""
    pass


def delay_late_starts(genome: np.ndarray) -> np.ndarray:
    """Move patients with late start times to the back of the schedule."""
    pass


# --- LARGE NEIGHBORHOOD ---


@njit
def destroy_path(genome: np.ndarray) -> np.ndarray:
    path_to_destroy = np.random.randint(0, genome.shape[0])
    genome[path_to_destroy, :] = 0
    return genome


@njit
def destroy_random(genome: np.ndarray, min: int, max: int) -> np.ndarray:
    """Destroy a random number of patients between min and max."""
    to_destroy = np.random.randint(min, max)
    patients = np.random.choice(genome.shape[1], to_destroy, replace=False)
    for patient in patients:
        patient_idx_ = np.where(genome == patient)
        patient_idx = (patient_idx_[0][0], patient_idx_[1][0])
        genome[patient_idx] = 0
    return genome


def destroy_violations(genome: np.ndarray) -> np.ndarray:
    to_destroy = []
    for nurse_path in genome:
        nurse_used_capacity = 0
        prev_spot_idx = 0

        cur_time = 0.0

        used_nurse_path = nurse_path[np.where(nurse_path != 0)]

        for patient_id in used_nurse_path:

            cur_time += problem.travel_times[prev_spot_idx, patient_id]

            if cur_time < problem.patients[patient_id - 1, 3]:
                cur_time = problem.patients[patient_id - 1, 3]
            elif cur_time > problem.patients[patient_id - 1, 4]:
                to_destroy.append(patient_id)

            cur_time += problem.patients[patient_id - 1, 5]
            nurse_used_capacity += problem.patients[patient_id - 1, 2]

            if cur_time > problem.patients[patient_id - 1, 4]:
                if patient_id not in to_destroy:
                    to_destroy.append(patient_id)

            prev_spot_idx = patient_id

        # penalize if capacity is exceeded
        if nurse_used_capacity > problem.capacity_nurse:
            if patient_id not in to_destroy:
                to_destroy.append(patient_id)

        # cur_time += problem.travel_times[prev_spot_idx, 0]
        # if cur_time > problem.depot_return_time:
        #     if patient_id not in to_destroy:
        #         to_destroy.append(patient_id)

    for patient_id in to_destroy:
        patient_idx_ = np.where(genome == patient_id)
        patient_idx = (patient_idx_[0][0], patient_idx_[1][0])
        genome[patient_idx] = 0

    return genome


def repair_random(genome: np.ndarray) -> np.ndarray:
    """Fill in missing patients at random spots."""
    n_patients = genome.shape[1]
    all_patients = np.arange(1, n_patients + 1)

    # method 1 of finding patients used
    patient_indices = np.where(genome != 0)
    used_patients = genome[patient_indices]
    # method 2 of finding patients used
    # used_patients = np.unique(genome).nonzero()[0]
    # find missing patients
    missing_patients = np.setdiff1d(all_patients, used_patients)
    # method 2 of finding missing patients
    # missing_patients = all_patients[~np.isin(all_patients, used_patients)]
    # fill in missing patients
    for patient in missing_patients:
        # select random nurse path
        nurse_path = np.random.randint(0, genome.shape[0])
        # try to find a spot in the nurse path to insert to
        idx = np.random.randint(0, genome.shape[1])
        while genome[nurse_path, idx] != 0:
            idx = np.random.randint(0, genome.shape[1])
        genome[nurse_path, idx] = patient

    return genome


def repair_random_uniform(genome: np.ndarray) -> np.ndarray:
    """Fill in missing patients at random spots with emphasis on nurses with few patients."""
    n_patients = genome.shape[1]
    all_patients = np.arange(1, n_patients + 1)

    # finding patients used
    patient_indices = np.where(genome != 0)
    used_patients = genome[patient_indices]
    # find missing patients
    missing_patients = np.setdiff1d(all_patients, used_patients)
    # fill in missing patients
    for patient in missing_patients:
        # find nurse path with fewest patients
        nurse_path_counts = np.count_nonzero(genome, axis=1)
        nurse_path = np.argmin(nurse_path_counts)
        # try to find a spot in the nurse path to insert to
        idx = np.random.randint(0, genome.shape[1])
        while genome[nurse_path, idx] != 0:
            idx = np.random.randint(0, genome.shape[1])
        genome[nurse_path, idx] = patient

    return genome


def repair_greedy(genome: np.ndarray) -> np.ndarray:
    """Fill in missing patients by greedily inserting them where there is least travel time."""
    n_patients = genome.shape[1]
    all_patients = np.arange(1, n_patients + 1)

    # finding patients used
    patient_indices = np.where(genome != 0)
    used_patients = genome[patient_indices]
    # find missing patients
    missing_patients = np.setdiff1d(all_patients, used_patients)
    # fill in missing patients
    for patient in missing_patients:
        # find nurse path with least travel time
        patients_by_distance = np.argsort(problem.travel_times[patient])
        # start with closest patient
        inserted = False
        # try to find a spot in the nurse path to insert to
        for patient_try in range(1, len(patients_by_distance)):
            # if nearest neighbor is also missing
            if not patients_by_distance[patient_try] in genome:
                continue
            spot_of_interest = np.where(genome == patients_by_distance[patient_try])
            # is free spot before
            if genome[spot_of_interest[0][0], spot_of_interest[1][0] - 1] == 0:
                genome[spot_of_interest[0][0], spot_of_interest[1][0] - 1] = patient
                inserted = True
                # roll nurse_path if inserted in [-1]
                if spot_of_interest[1][0] == 0:
                    genome[spot_of_interest[0][0]] = np.roll(
                        genome[spot_of_interest[0][0]], 1
                    )
                break
            # is free spot after and in last position
            if spot_of_interest[1][0] == genome.shape[1] - 1:
                if genome[spot_of_interest[0][0], 0] == 0:
                    genome[spot_of_interest[0][0], 0] = patient
                    inserted = True
                    # roll nurse_path
                    genome[spot_of_interest[0][0]] = np.roll(
                        genome[spot_of_interest[0][0]], -1
                    )
                    break
            else:
                # is free spot after
                if genome[spot_of_interest[0][0], spot_of_interest[1][0] + 1] == 0:
                    genome[spot_of_interest[0][0], spot_of_interest[1][0] + 1] = patient
                    inserted = True
                    break
        # if no spot found, insert at random
        if not inserted:
            # select random nurse path
            nurse_path = np.random.randint(0, genome.shape[0])
            # try to find a spot in the nurse path to insert to
            idx = np.random.randint(0, genome.shape[1])
            while genome[nurse_path, idx] != 0:
                idx = np.random.randint(0, genome.shape[1])
            genome[nurse_path, idx] = patient

    return genome


# --- LOCAL SEARCH ---
# permorm some mutations, evaluate and keep if better


def small_local_search(genome: np.ndarray, iterations: int) -> np.ndarray:
    """Perform some mutations, evaluate and keep if better reset if worse."""
    original_fitness = evaluate(genome)
    for _ in range(iterations):
        genome_ = genome.copy()
        mut_type = np.random.randint(0, 13)
        if mut_type == 0:
            genome_ = shuffle_path(genome)
        elif mut_type == 1:
            genome_ = random_multi_swap(genome)
        elif mut_type == 2:
            genome_ = close_single_swap(genome, m=4)
        elif mut_type == 3:
            genome_ = far_single_swap(genome, m=4)
        elif mut_type == 4:
            genome_ = close_single_swap(genome, m=2)
            genome_ = far_single_swap(genome_, m=2)
        elif mut_type == 5:
            genome_ = destroy_violations(genome)
            genome_ = repair_greedy(genome_)
        elif mut_type == 6:
            genome_ = destroy_violations(genome)
            genome_ = repair_random(genome_)
        elif mut_type == 7:
            genome_ = destroy_violations(genome)
            genome_ = repair_random_uniform(genome_)
        elif mut_type == 8:
            genome_ = destroy_random(genome, 1, 6)
            genome_ = repair_greedy(genome_)
        elif mut_type == 9:
            genome_ = destroy_random(genome, 1, 6)
            genome_ = repair_random_uniform(genome_)
        elif mut_type == 10:
            genome_ = destroy_random(genome, 1, 6)
            genome_ = repair_random(genome_)
        elif mut_type == 11:
            genome_ = destroy_path(genome)
            genome_ = repair_greedy(genome_)
        elif mut_type == 12:
            genome_ = destroy_path(genome)
            genome_ = repair_random_uniform(genome_)
        fitness = evaluate(genome_)
        if fitness < original_fitness:
            genome[:] = genome_[:]
            original_fitness = fitness
    return genome


@njit
def large_local_search(genome: np.ndarray, iterations: int) -> np.ndarray:
    """Perform some mutations, evaluate and keep if better don't reset if worse."""
    original_fitness = evaluate(genome)
    genome_ = genome.copy()
    for _ in range(iterations):
        mut_type = np.random.randint(0, 7)
        if mut_type == 0:
            genome_ = shuffle_path(genome)
        elif mut_type == 1:
            genome_ = random_multi_swap(genome)
        elif mut_type == 2:
            genome_ = close_single_swap(genome, m=4)
        elif mut_type == 3:
            genome_ = far_single_swap(genome, m=4)
        elif mut_type == 4:
            genome_ = close_single_swap(genome, m=2)
            genome_ = far_single_swap(genome_, m=2)
        fitness = evaluate(genome_)
        if fitness < original_fitness:
            genome[:] = genome_[:]
            original_fitness = fitness
    return genome


# --- COMBINED MUTATIONS ---


def mutate_population(population: np.ndarray, m: int) -> np.ndarray:
    """Mutate a population of genomes."""
    # genome_size = population.shape[1]
    for idx, genome in enumerate(population):
        # select mutation type
        mut_type = np.random.randint(0, 12)
        if mut_type == 0:
            genome_ = small_local_search(genome, iterations=8)
        elif mut_type == 1:
            genome_ = roll_path(genome)
        elif mut_type == 2:
            genome_ = shuffle_path(genome)
        elif mut_type == 3:
            genome_ = flip_path(genome)
        elif mut_type == 4:
            genome_ = random_multi_swap(genome)
        elif mut_type == 5:
            genome_ = close_single_swap(genome, m=m)
        elif mut_type == 6:
            genome_ = far_single_swap(genome, m=m)
        elif mut_type == 7:
            genome_ = small_local_search(genome, iterations=14)
        elif mut_type == 8:
            genome_ = large_local_search(genome, iterations=10)
        elif mut_type == 9:
            genome_ = destroy_violations(genome)
            genome_ = repair_greedy(genome_)
        elif mut_type == 10:
            genome_ = destroy_violations(genome)
            genome_ = repair_random_uniform(genome_)
        else:
            genome_ = genome
        population[idx, :] = genome_
    return population


# --- TIMING ---
def timing():
    genome = generate_random_genome(n_patients=100, n_nurses=25)

    for _ in tqdm(range(100_000)):
        #     # genome = intra_move_random(genome)
        #     intra_move_swap(genome)
        #     roll_path(genome)
        #     shuffle_path(genome)
        #     flip_path(genome)
        #     index_interval_inter_swap(genome)
        genome = close_single_swap(genome, m=8)
        genome = far_single_swap(genome, m=8)


def timing2():
    population = generate_random_population(size=50, n_patients=100, n_nurses=25)

    for _ in tqdm(range(100_000)):
        # genome = intra_move_random(genome)
        population = mutate_population(population)


if __name__ == "__main__":
    from initializations import generate_random_genome, generate_random_population
    import cProfile, pstats
    from tqdm import tqdm

    profiler = cProfile.Profile()
    profiler.enable()

    timing()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("profile/mutations.prof")
