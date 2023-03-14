import json
import time
import itertools

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from numba import njit


class Problem:
    """A problem class that loads from json file,
    handles visualization and evaluation of solutions."""

    def __init__(self, file: str) -> None:
        self.file = file
        with open(file, "r", encoding="UTF-8") as file:
            self.data = json.load(file)

        self.instance_name = self.data["instance_name"]
        self.nbr_nurses = self.data["nbr_nurses"]
        self.capacity_nurse = self.data["capacity_nurse"]
        self.depot = self.data["depot"]
        self.patients = self.data["patients"]
        self.numpy_patients = self.patients_to_numpy(self.patients)
        # assumes id 1-len(patients) are used
        self.nbr_patients = len(self.patients)
        self.travel_times = np.asarray(self.data["travel_times"])

    def patients_to_numpy(self, patients: dict[str, dict[str, int]]) -> NDArray:
        """Converts patients to numpy array."""
        patients = [
            [
                patient["x_coord"],
                patient["y_coord"],
                patient["demand"],
                patient["start_time"],
                patient["end_time"],
                patient["care_time"],
            ]
            for patient in patients.values()
        ]
        return np.asarray(patients)

    def visualize_problem(self) -> None:
        """Visualize the problem instance."""
        # plot depot
        plt.scatter(self.depot["x_coord"], self.depot["y_coord"], c="r")
        # plot patients
        for idx in range(1, self.nbr_patients):
            idx = str(idx)
            plt.scatter(
                self.patients[idx]["x_coord"], self.patients[idx]["y_coord"], c="b"
            )
        plt.show()

    def visualize_solution(self, solution: list | NDArray) -> None:
        """Visualize a solution.

        Args:
            solution (list | NDArray): Solution on either list on numpy format.
        """

        if isinstance(solution, list):
            list_solution = solution
        else:
            list_solution = solution_to_list(solution)

        # plot depot
        plt.scatter(self.depot["x_coord"], self.depot["y_coord"], c="r")
        # plot patients
        for idx in range(1, self.nbr_patients):
            idx = str(idx)
            plt.scatter(
                self.patients[idx]["x_coord"], self.patients[idx]["y_coord"], c="b"
            )
        # plot solution routes
        for nurse in list_solution:
            # add depot to start and end of route
            xs = [self.depot["x_coord"]]
            ys = [self.depot["y_coord"]]
            # add patients to route
            for patient in nurse:
                idx = str(patient)
                xs.append(self.patients[idx]["x_coord"])
                ys.append(self.patients[idx]["y_coord"])

            xs.append(self.depot["x_coord"])
            ys.append(self.depot["y_coord"])
            plt.plot(xs, ys)

        plt.show()

    def print_solution(self, solution: list | NDArray) -> None:
        """Prints a solution on the desired format.

        Args:
            solution (list | NDArray): Solution on either list on numpy format.
        """

        if isinstance(solution, list):
            list_solution = solution
            numpy_solution = solution_to_numpy(solution)
        else:
            list_solution = solution_to_list(solution)
            numpy_solution = solution

        print()
        print("SOLUTION")
        print(f"Nurse capacity: {self.capacity_nurse}")
        print()
        print(f"Depot return time: {self.depot['return_time']}")
        print("------------------------------------------------")
        # check nurse path
        for nurse_idx, nurse_patients in enumerate(list_solution):
            # calculate used nurse capacity
            nurse_used_capacity = 0
            # calculate time
            tot_time = 0
            # add depot to start
            prev_spot_idx = 0
            # setup patient sequence, with start used demand used 0
            patient_sequence = ["D(0)"]
            # add patients to route
            for patient in nurse_patients:
                str_idx = str(patient)

                # get travel time
                travel_time = self.travel_times[prev_spot_idx, patient]
                tot_time += travel_time
                arrival_time = tot_time

                # check if time window is met
                # penalty is both added if arrival after end time and if service ends after end time
                if tot_time < self.patients[str_idx]["start_time"]:
                    # wait until start time
                    tot_time = self.patients[str_idx]["start_time"]

                # add service time
                tot_time += self.patients[str_idx]["care_time"]
                # add used capacity
                nurse_used_capacity += self.patients[str_idx]["demand"]

                # update prev spot
                prev_spot_idx = patient

                # add patient to sequence
                patient_visit_info = (
                    f"{patient} ({arrival_time:.2f}-{arrival_time + self.patients[str_idx]['care_time']:.2f})"
                    + f" [{self.patients[str_idx]['start_time']:.2f}-{self.patients[str_idx]['end_time']:.2f}]"
                )
                patient_sequence.append(patient_visit_info)
                # add current used demand
                patient_sequence.append(f"D({nurse_used_capacity})")

            print(
                f"Nurse {nurse_idx}"
                + f" | {tot_time}"
                + f" | {nurse_used_capacity}"
                + f" | {patient_sequence}"
            )
            # space inbetween nurses
            print()

        print("------------------------------------------------")

        objective_value, is_valid = evaluate(
            solution=numpy_solution,
            travel_times=self.travel_times,
            capacity_nurse=self.capacity_nurse,
            patients=self.numpy_patients,
            penalize_invalid=False,
        )
        print(f"Objective value (total duration): {objective_value}")
        print(f"Valid solution: {is_valid}")


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
def evaluate(
    solution: NDArray,
    travel_times: NDArray,
    capacity_nurse: int,
    patients: NDArray,
    penalize_invalid: bool = False,
) -> float:
    """Evalauates a solution.

    Args:
        solution (NDArray): A potential solution.
        Solution is on format one row per nurse,
        with id for each patient visisted,
        filled with zeros inbetween.
        travel_times (NDArray): _description_
        capacity_nurse (int): _description_
        patients (NDArray): _description_
        penalize_invalid (bool, optional): _description_. Defaults to False.

    Returns:
        float: fitness of the solution.
    """

    fitness = 0.0
    is_valid = True
    # assumes all patients are visited

    # check nurse path
    for nurse_path in solution:
        # calculate used nurse capacity
        nurse_used_capacity = 0
        # calculate time
        tot_time = 0.0
        # add depot to start
        prev_spot_idx = 0

        # get used nurse_path
        used_nurse_path = nurse_path[np.where(nurse_path != 0)]

        # add patients to route
        for patient_id in used_nurse_path:

            tot_time += travel_times[prev_spot_idx, patient_id]

            # check if time window is met
            # penalty is both added if arrival after end time and if service ends after end time
            if tot_time < patients[patient_id - 1, 3]:
                # wait until start time
                tot_time = patients[patient_id - 1, 3]
            elif tot_time > patients[patient_id - 1, 4]:
                # penalize if time window is not met
                if penalize_invalid:
                    fitness += 1000
                is_valid = False

            # add service time
            tot_time += patients[patient_id - 1, 5]
            # add used capacity
            nurse_used_capacity += patients[patient_id - 1, 2]

            # penalize if time is after end time
            if tot_time > patients[patient_id - 1, 4]:
                if penalize_invalid:
                    fitness += 1000
                is_valid = False

            # update prev spot
            prev_spot_idx = patient_id

        # penalize if capacity is exceeded
        if nurse_used_capacity > capacity_nurse:
            if penalize_invalid:
                fitness += 1000
            is_valid = False

        # add time to fitness
        fitness += tot_time

    return fitness, is_valid


@njit
def generate_random_solution(n_nurses: int, n_patients: int) -> NDArray:
    """Generate a random solution."""
    patient_ids = np.arange(1, n_patients + 1)
    solution = np.zeros((n_nurses * n_patients), dtype=np.int32)

    # Choose random indices to insert values from
    indices = np.random.choice(
        np.arange(n_nurses * n_patients), size=n_patients, replace=False
    )

    # Insert values from patient_ids into zeros
    solution[indices] = patient_ids

    # Reshape to (n_nurses, n_patients)
    solution = solution.reshape((n_nurses, n_patients))

    return solution


def main():
    # Load the json file
    problem = Problem("data/train_0.json")
    print(f"instance_name {problem.instance_name}")
    print(f"nbr_nurses {problem.nbr_nurses}")
    print(f"capacity_nurse {problem.capacity_nurse}")
    print(f"depot {problem.depot}")
    print(f"patients {problem.patients}")
    print(f"travel_times {problem.travel_times}")
    print(f"nbr_patients {problem.nbr_patients}")
    print(f"numpy_patients {problem.numpy_patients}")
    # problem.visualize_problem()

    # generate random solution
    sol = generate_random_solution(
        n_nurses=problem.nbr_nurses, n_patients=problem.nbr_patients
    )
    fitness, is_valid = evaluate(
        solution=sol,
        travel_times=problem.travel_times,
        capacity_nurse=problem.capacity_nurse,
        patients=problem.numpy_patients,
    )
    list_sol = solution_to_list(sol)
    numpy_sol = solution_to_numpy(list_sol)
    problem.print_solution(numpy_sol)
    problem.visualize_solution(list_sol)

    # Run the GA
    # ga = GA(problem, data)
    # ga.run()


def eval_timing():
    problem = Problem("data/train_0.json")
    # evaluate running timing of solution
    numba_times = []
    gen_times = []
    for _ in range(10_000):
        before1 = time.perf_counter_ns()
        sol = generate_random_solution(
            n_nurses=problem.nbr_nurses, n_patients=problem.nbr_patients
        )
        after1 = time.perf_counter_ns()
        generation_time = after1 - before1

        before2 = time.perf_counter_ns()

        _, _ = numba_eval(
            solution=sol,
            travel_times=problem.travel_times,
            capacity_nurse=problem.capacity_nurse,
            patients=problem.numpy_patients,
            penalize_invalid=False,
        )

        after2 = time.perf_counter_ns()
        numba_time = after2 - before2

        gen_times.append(generation_time)
        numba_times.append(numba_time)

    print(f"Average generation time: {np.mean(gen_times[1:])} ns")
    print(f"Average numba time: {np.mean(numba_times[1:])} ns")


if __name__ == "__main__":
    main()
