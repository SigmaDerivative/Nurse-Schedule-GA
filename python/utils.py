import numpy as np
from numba import njit


def entropy():
    pass
    # bitstr_numbers = np.zeros(bitstr_size)
    # # adds together all genomes for every candidate
    # for x in pop:
    #     bitstr_numbers += x.bitstring

    # # convert to probability for each genome
    # H = bitstr_numbers / len(pop)

    # # calculate entropy, do not add 0 values
    # H = -np.sum(np.where(H == 0, 0, H * np.log2(H)))

    # return H
