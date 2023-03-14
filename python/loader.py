import json
import numpy as np
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r", encoding="UTF-8") as file:
        return json.load(file)


class Problem:
    def __init__(self, file: str) -> None:
        self.file = file
        self.data = load_json(file)
        self.instance_name = self.data["instance_name"]
        self.nbr_nurses = self.data["nbr_nurses"]
        self.capacity_nurse = self.data["capacity_nurse"]
        self.depot = self.data["depot"]
        self.patients = self.data["patients"]
        self.nbr_patients = len(self.patients)
        self.travel_times = np.asarray(self.data["travel_times"])

    def visualize(self):
        # plot depot
        plt.scatter(self.depot["x_coord"], self.depot["y_coord"], c="r")
        # plot patients
        for idx in range(1, self.nbr_patients):
            idx = str(idx)
            plt.scatter(
                self.patients[idx]["x_coord"], self.patients[idx]["y_coord"], c="b"
            )
        plt.show()


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
    problem.visualize()

    # Run the GA
    # ga = GA(problem, data)
    # ga.run()
