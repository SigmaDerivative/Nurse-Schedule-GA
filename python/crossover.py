def parent_selection(population: list, num_parents: int) -> list:
    # sort population
    population.sort(key=lambda x: x.fitness)
    # return the two first
    return population[:num_parents]


def crossover(self):
    pass


def mate(self):
    pass


def duel(self):
    pass
