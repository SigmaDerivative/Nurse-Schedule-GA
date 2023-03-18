import matplotlib.pyplot as plt
import numpy as np


def plot_fitness(name: str, num_ep: int, avg_fit: np.ndarray):
    plt.clf()
    plt.plot(np.arange(num_ep), avg_fit, label="fitness")
    plt.legend()
    plt.savefig(name)


# def plot_entropy(name: str, num_ep: int, entropies: np.ndarray):
#     plt.clf()
#     plt.plot(
#         np.arange(num_ep),
#         entropies,
#         c="r",
#         label="entropy",
#     )
#     plt.legend()
#     plt.savefig(name)
