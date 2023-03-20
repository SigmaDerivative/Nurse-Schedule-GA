from numba import njit
import numpy as np

import problem


@njit
def evaluate(
    solution: np.ndarray,
    penalize_invalid: bool = True,
) -> float:
    """Evalauates a solution.

    Args:
        solution (np.ndarray): A potential solution.
        Solution is on format one row per nurse,
        with id for each patient visisted,
        filled with zeros inbetween.

        penalize_invalid (bool, optional): If fitness is penalized by invalid solution. Defaults to True.

    Returns:
        tuple[float, bool]: fitness, is_valid
    """
    fitness = 0.0
    is_valid = True
    # assumes all patients are visited

    # check nurse path
    for nurse_path in solution:
        # calculate used nurse capacity
        nurse_used_capacity = 0
        # calculate time
        travel_time = 0.0
        # add depot to start
        prev_spot_idx = 0

        # counter for current time
        cur_time = 0.0

        # get used nurse_path
        used_nurse_path = nurse_path[np.where(nurse_path != 0)]

        # add patients to route
        for patient_id in used_nurse_path:

            travel_time += problem.travel_times[prev_spot_idx, patient_id]
            cur_time += problem.travel_times[prev_spot_idx, patient_id]

            # check if time window is met
            # penalty is both added if arrival after end time and if service ends after end time
            if cur_time < problem.patients[patient_id - 1, 3]:
                # wait until start time
                cur_time = problem.patients[patient_id - 1, 3]
            elif cur_time > problem.patients[patient_id - 1, 4]:
                # penalize if time window is not met
                if penalize_invalid:
                    fitness += (
                        cur_time - problem.patients[patient_id - 1, 4]
                    ) * problem.start_after_end_penalty
                is_valid = False

            # add service time
            cur_time += problem.patients[patient_id - 1, 5]
            # add used capacity
            nurse_used_capacity += problem.patients[patient_id - 1, 2]

            # penalize if time is after end time
            if cur_time > problem.patients[patient_id - 1, 4]:
                if penalize_invalid:
                    fitness += (
                        cur_time - problem.patients[patient_id - 1, 4]
                    ) * problem.end_after_end_penalty
                is_valid = False

            # update prev spot
            prev_spot_idx = patient_id

        # penalize if capacity is exceeded
        if nurse_used_capacity > problem.capacity_nurse:
            if penalize_invalid:
                fitness += (
                    nurse_used_capacity - problem.capacity_nurse
                ) * problem.capacity_penalty
            is_valid = False

        # add depot to end
        travel_time += problem.travel_times[prev_spot_idx, 0]
        cur_time += problem.travel_times[prev_spot_idx, 0]
        # penalize if depot return time not met
        if cur_time > problem.depot_return_time:
            if penalize_invalid:
                fitness += (
                    cur_time - problem.depot_return_time
                ) * problem.start_after_end_penalty
            is_valid = False
        # add time to fitness
        fitness += travel_time

    return fitness, is_valid


@njit
def evaluate_population(
    genomes: np.ndarray, penalize_invalid: bool = True
) -> np.ndarray:
    fitness = np.empty((genomes.shape[0], 1), dtype=np.float64)
    valid = np.empty((genomes.shape[0], 1), dtype=np.bool_)
    for idx, genome in enumerate(genomes):
        fitness_, valid_ = evaluate(genome, penalize_invalid=penalize_invalid)
        fitness[idx, 0] = fitness_
        valid[idx, 0] = valid_
    return fitness, valid
