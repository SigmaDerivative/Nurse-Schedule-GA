import json
import time

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


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
        # assumes id 1-len(patients) are used
        self.nbr_patients = len(self.patients)
        self.travel_times = np.asarray(self.data["travel_times"])

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

    def evaluate(
        self, solution: NDArray, penalize_invalid: bool = False, timing: bool = False
    ) -> float:
        """Evalauates a solution.

        Args:
            solution (NDArray): A potential solution.
            Solution is on format one row per nurse, with id for each patient visisted.

        Returns:
            float: fitness of the solution.
        """
        if timing:
            start = time.time()

        fitness = 0.0
        is_valid = True
        # assumes all patients are visited

        # check nurse path
        for nurse in solution:
            # calculate used nurse capacity
            nurse_used_capacity = 0
            # calculate time
            tot_time = 0
            # add depot to start
            prev_spot_idx = 0
            # add patients to route
            for patient in nurse:
                str_idx = str(patient)

                tot_time += self.travel_times[prev_spot_idx, patient]

                # check if time window is met
                # penalty is both added if arrival after end time and if service ends after end time
                if tot_time < self.patients[str_idx]["start_time"]:
                    # wait until start time
                    tot_time = self.patients[str_idx]["start_time"]
                elif tot_time > self.patients[str_idx]["end_time"]:
                    # penalize if time window is not met
                    if penalize_invalid:
                        fitness += 1000
                    is_valid = False

                # add service time
                tot_time += self.patients[str_idx]["care_time"]
                # add used capacity
                nurse_used_capacity += self.patients[str_idx]["demand"]

                # penalize if time is after end time
                if tot_time > self.patients[str_idx]["end_time"]:
                    if penalize_invalid:
                        fitness += 1000
                    is_valid = False

                # update prev spot
                prev_spot_idx = patient

            # penalize if capacity is exceeded
            if nurse_used_capacity > self.capacity_nurse:
                if penalize_invalid:
                    fitness += 1000
                is_valid = False

            # add time to fitness
            fitness += tot_time

        if timing:
            print(f"Evaluation time: {time.time() - start:.2f} seconds")

        return fitness, is_valid

    def visualize_solution(self, solution: NDArray) -> None:
        """Visualize a solution."""
        # plot depot
        plt.scatter(self.depot["x_coord"], self.depot["y_coord"], c="r")
        # plot patients
        for idx in range(1, self.nbr_patients):
            idx = str(idx)
            plt.scatter(
                self.patients[idx]["x_coord"], self.patients[idx]["y_coord"], c="b"
            )
        # plot solution routes
        for nurse in solution:
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

    def print_solution(self, solution: NDArray) -> None:
        """Prints a solution on the desired format."""

        print()
        print("SOLUTION")
        print(f"Nurse capacity: {self.capacity_nurse}")
        print()
        print(f"Depot return time: {self.depot['return_time']}")
        print("------------------------------------------------")
        # check nurse path
        for nurse_idx, nurse_patients in enumerate(solution):
            # calculate used nurse capacity
            nurse_used_capacity = 0
            # calculate time
            tot_time = 0
            # add depot to start
            prev_spot_idx = 0
            # setup patient sequence, with start used demand
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
                    f"{patient} ({arrival_time:.2f}-{self.patients[str_idx]['care_time']:.2f})"
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

        objective_value, is_valid = self.evaluate(solution, penalize_invalid=False)

        print("------------------------------------------------")
        print(f"Objective value (total duration): {objective_value}")
        print(f"Valid solution: {is_valid}")


def generate_random_solution(problem: Problem) -> NDArray:
    """Generate a random solution."""
    patient_ids = list(problem.patients.keys())
    values = [[] for _ in range(problem.nbr_nurses)]
    while patient_ids:
        nurse = np.random.randint(0, problem.nbr_nurses)
        values[nurse].append(int(patient_ids.pop()))
    # return as numpy array
    return [np.asarray(i) for i in values]


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
    # problem.visualize_problem()

    # generate random solution
    sol = generate_random_solution(problem)
    problem.print_solution(sol)
    problem.visualize_solution(sol)

    # Run the GA
    # ga = GA(problem, data)
    # ga.run()


if __name__ == "__main__":
    main()
