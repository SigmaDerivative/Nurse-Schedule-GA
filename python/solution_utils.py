import itertools

import numpy as np
from numpy.typing import NDArray
from numba import njit


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
