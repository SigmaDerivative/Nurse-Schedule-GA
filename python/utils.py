import itertools

import numpy as np


def entropy():
    pass


def solution_to_numpy(solution: list) -> np.ndarray:
    """Converts a solution to a numpy array.

    Code from: https://stackoverflow.com/questions/38619143/convert-python-sequence-to-numpy-array-filling-missing-values

    Args:
        solution (list): A potential solution.
        Solution is on format one row per nurse,
        with id for each patient visisted.

    Returns:
        np.ndarray: A numpy array, where each row is a nurse route.
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


def solution_to_list(solution: np.ndarray) -> list:
    """Converts a solution to a list of lists.

    Args:
        solution (np.ndarray): A potential solution.
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


def insert_inbetween():
    """Moves one patient index inbetween two other patient indices."""
    pass
