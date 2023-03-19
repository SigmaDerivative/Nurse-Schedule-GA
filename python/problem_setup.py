import json
import time

import numpy as np
import matplotlib.pyplot as plt

from utils import solution_to_numpy, solution_to_list


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
        self.list_patients = self.data["patients"]
        self.numpy_patients = self.list_patients_to_numpy(self.list_patients)
        # assumes id 1-len(patients) are used
        self.nbr_patients = len(self.numpy_patients)
        self.travel_times = np.asarray(self.data["travel_times"])

    def list_patients_to_numpy(
        self, list_patients: dict[str, dict[str, int]]
    ) -> np.ndarray:
        """Converts patients to numpy array."""
        list_patients = [
            [
                patient["x_coord"],
                patient["y_coord"],
                patient["demand"],
                patient["start_time"],
                patient["end_time"],
                patient["care_time"],
            ]
            for patient in list_patients.values()
        ]
        return np.asarray(list_patients)

    def visualize_problem(self) -> None:
        """Visualize the problem instance."""
        # plot depot
        plt.scatter(self.depot["x_coord"], self.depot["y_coord"], c="r")
        # plot patients
        for idx in range(1, self.nbr_patients):
            idx = str(idx)
            plt.scatter(
                self.list_patients[idx]["x_coord"],
                self.list_patients[idx]["y_coord"],
                c="b",
            )
        plt.show()

    def visualize_solution(self, solution: list | np.ndarray) -> None:
        """Visualize a solution.

        Args:
            solution (list | np.ndarray): Solution on either list on numpy format.
        """

        if isinstance(solution, list):
            list_solution = solution
        else:
            list_solution = solution_to_list(solution)

        # plot depot
        plt.scatter(self.depot["x_coord"], self.depot["y_coord"], c="r")
        # plot patients
        for idx in range(1, self.nbr_patients + 1):
            idx = str(idx)
            plt.scatter(
                self.list_patients[idx]["x_coord"],
                self.list_patients[idx]["y_coord"],
                c="b",
            )
        # plot solution routes
        for nurse in list_solution:
            # add depot to start and end of route
            xs = [self.depot["x_coord"]]
            ys = [self.depot["y_coord"]]
            # add patients to route
            for patient in nurse:
                idx = str(patient)
                xs.append(self.list_patients[idx]["x_coord"])
                ys.append(self.list_patients[idx]["y_coord"])

            xs.append(self.depot["x_coord"])
            ys.append(self.depot["y_coord"])
            plt.plot(xs, ys)

        plt.show()

    def print_solution(
        self, solution: list | np.ndarray, fitness: float, is_valid: bool
    ) -> None:
        """Prints a solution on the desired format.

        Args:
            solution (list | np.ndarray): Solution on either list on numpy format.
        """

        if isinstance(solution, list):
            list_solution = solution
            # numpy_solution = solution_to_numpy(solution)
        else:
            list_solution = solution_to_list(solution)
            # numpy_solution = solution

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
                if tot_time < self.list_patients[str_idx]["start_time"]:
                    # wait until start time
                    tot_time = self.list_patients[str_idx]["start_time"]

                # add service time
                tot_time += self.list_patients[str_idx]["care_time"]
                # add used capacity
                nurse_used_capacity += self.list_patients[str_idx]["demand"]

                # update prev spot
                prev_spot_idx = patient

                # add patient to sequence
                patient_visit_info = (
                    f"{patient} ({arrival_time:.2f}-{arrival_time + self.list_patients[str_idx]['care_time']:.2f})"
                    + f" [{self.list_patients[str_idx]['start_time']:.2f}-{self.list_patients[str_idx]['end_time']:.2f}]"
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
        print(f"Objective value (total duration): {fitness}")
        print(f"Valid solution: {is_valid}")


if __name__ == "__main__":
    from initializations import generate_random_genome
    from evaluations import evaluate

    problem = Problem("data/train_0.json")
    print(f"instance_name {problem.instance_name}")
    print(f"nbr_nurses {problem.nbr_nurses}")
    print(f"capacity_nurse {problem.capacity_nurse}")
    print(f"depot {problem.depot}")
    print(f"patients {problem.numpy_patients}")
    print(f"travel_times {problem.travel_times}")
    print(f"nbr_patients {problem.nbr_patients}")
    print(f"list_patients {problem.list_patients}")
    print(f"numpy_patients {problem.numpy_patients}")

    sol = generate_random_genome(
        n_nurses=problem.nbr_nurses, n_patients=problem.nbr_patients
    )
    fitness, is_valid = evaluate(
        solution=sol,
    )
    problem.print_solution(sol, fitness, is_valid)
    problem.visualize_solution(sol)
