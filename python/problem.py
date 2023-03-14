import json
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

    def evaluate(self, solution: NDArray) -> float:
        """Evalauates a solution.

        Args:
            solution (NDArray): A potential solution.
            Solution is on format one row per nurse, one hot encoded for each patient visisted.

        Returns:
            float: fitness of the solution.
        """
        # TODO
        return 0.0

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
        # TODO plot solution routes


if __name__ == "__main__":
    # Load the json file
    problem = Problem("data/train_0.json")
    print(problem.instance_name)
    print(problem.nbr_nurses)
    print(problem.capacity_nurse)
    print(problem.depot)
    print(problem.patients)
    print(problem.travel_times)
    print(problem.nbr_patients)
    problem.visualize_problem()

    # Run the GA
    # ga = GA(problem, data)
    # ga.run()
