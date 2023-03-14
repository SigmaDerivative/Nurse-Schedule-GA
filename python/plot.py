import matplotlib.pyplot as plt
import numpy as np

from ga import Individual


def plot(
    maximize: bool, name: str, num_ep: int, avg_fit: list[float], pop: list[Individual]
):
    plt.clf()
    if maximize:
        plt.plot(np.arange(num_ep), avg_fit, label="RMSE feature selected")
        plt.plot(
            np.arange(num_ep),
            np.ones(num_ep),  # TODO * fitness
            c="r",
            label="base RMSE",
        )
        # plt.plot(np.arange(num_ep), np.ones(num_ep) * 0.124, c="g", label="0.124")
        plt.legend()
    else:
        plt.plot(np.arange(0, 128, 0.5), np.sin(np.arange(0, 128, 0.5)))
        for i in pop:
            plt.scatter(i.get_solution(), np.sin(i.get_solution()), c="r")
    plt.savefig(name)


def plot_entropy(name: str, num_ep: int, entropies: list[float]):
    plt.clf()
    plt.plot(
        np.arange(num_ep),
        entropies,
        c="r",
        label="entropy",
    )
    plt.legend()
    plt.savefig(name)
