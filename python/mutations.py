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


# --- GREEDY MUTATIONS ---


# --- TIMING ---
def timing():
    genome = generate_random_solution(n_patients=100, n_nurses=25)

    for _ in tqdm(range(1_000_000)):
        # genome = intra_move_random(genome)
        intra_move_swap(genome)
        roll_path(genome)
        shuffle_path(genome)
        flip_path(genome)


if __name__ == "__main__":
    from solution_utils import generate_random_solution
    import cProfile, pstats

    profiler = cProfile.Profile()
    profiler.enable()

    timing()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("profile/mutations.prof")
