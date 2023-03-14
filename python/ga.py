import time

import numpy as np


class Individual:
    def __init__(self, bitstr_size: int, maximize: bool) -> None:
        # inverse binary repr
        self.bitstring: np.ndarray[int] = np.random.randint(0, 2, bitstr_size)
        self.evaluate(maximize=maximize)

    def get_solution(self) -> int:
        """Gives the integer representation of the bitstring

        Returns:
            int: Integer to pass to sine function
        """
        num = 0
        for idx, bit_num in enumerate(self.bitstring):
            num += 2**idx * bit_num
        return num / (2 ** (len(self.bitstring) - 7))

    def evaluate(self, maximize: bool) -> float:
        # set fitness attribute
        fitness = 0
        self.fitness = fitness

    def mutate(
        self,
        genome_mutate_chance: float,
        min_mutate_num: int,
        max_mutate_num: int,
    ) -> None:
        # number of genomes to pick mutate
        num_genomes_to_mutate = np.random.randint(min_mutate_num, max_mutate_num + 1)
        # select random places in bitstring
        genome_indices = np.random.choice(
            np.arange(len(self.bitstring)), num_genomes_to_mutate, replace=False
        )
        # invert bitstring in mutated spots
        for idx in genome_indices:
            if np.random.uniform() < genome_mutate_chance:
                self.bitstring[idx] = 1 - self.bitstring[idx]


def create_population(size: int, bitstr_size: int, maximize: bool) -> list[Individual]:
    return [Individual(bitstr_size=bitstr_size, maximize=maximize) for _ in range(size)]


def parent_selection(
    population: list[Individual], num_parents: int, maximize: bool
) -> list[Individual]:
    if num_parents % 2 != 0:
        raise ValueError("num_parents not dividable by 2")

    # sort population
    if maximize:
        # get the worst
        population.sort(key=lambda x: x.fitness)
    else:
        # get the best
        population.sort(key=lambda x: x.fitness, reverse=True)
    # return the two first
    return population[:num_parents]


def crossover(
    parents: np.ndarray[Individual],
    bitstr_size: int,
    maximize: bool,
    num_children: int,
    deterministic: bool,
    ranked: bool,
) -> np.ndarray[Individual]:

    children = np.array([])

    if deterministic and num_children > len(parents):
        raise ValueError(
            "Cannot choose determinisitc parents to create more children than parents"
        )
    elif deterministic:
        for idx in range(int(len(parents) / 2)):
            pair = np.array([parents[idx * 2], parents[idx * 2 + 1]])
            children = np.append(
                children,
                mate(
                    pair=pair,
                    mutate_chance=0.5,
                    bitstr_size=bitstr_size,
                    maximize=maximize,
                ),
            )
    else:
        for _ in range(int(num_children / 2)):
            if ranked:
                prob = np.array([x.fitness + 1 for x in parents])
                prob = prob / np.sum(prob)
                pair = np.random.choice(parents, 2, replace=False, p=prob)
            else:
                pair = np.random.choice(parents, 2, replace=False)
            children = np.append(
                children,
                mate(
                    pair=pair,
                    mutate_chance=0.5,
                    bitstr_size=bitstr_size,
                    maximize=maximize,
                ),
            )

    return children


def mate(
    pair: np.ndarray[Individual], mutate_chance: float, bitstr_size: int, maximize: bool
) -> np.ndarray[Individual]:
    # find point to split where to get dna from parents
    split_index = np.random.randint(1, bitstr_size - 1)
    # create child1 and insert dna
    child1 = Individual(bitstr_size=bitstr_size, maximize=maximize)
    child1.bitstring = np.append(
        pair[0].bitstring[:split_index], pair[1].bitstring[split_index:]
    )
    # create child2 and insert dna
    child2 = Individual(bitstr_size=bitstr_size, maximize=maximize)
    child2.bitstring = np.append(
        pair[1].bitstring[:split_index], pair[0].bitstring[split_index:]
    )
    # mutate children
    if np.random.uniform() < mutate_chance:
        child1.mutate(
            genome_mutate_chance=0.5, min_mutate_num=1, max_mutate_num=bitstr_size
        )
    if np.random.uniform() < mutate_chance:
        child2.mutate(
            genome_mutate_chance=0.5, min_mutate_num=1, max_mutate_num=bitstr_size
        )

    child1.evaluate(maximize=maximize)
    child2.evaluate(maximize=maximize)

    children = np.array([child1, child2])

    return children


def tournament(
    population: list[Individual], pop_size: int, maximize: bool
) -> list[Individual]:
    # sort population
    if maximize:
        # the worst survive
        population.sort(key=lambda x: x.fitness)
    else:
        # the best survive
        population.sort(key=lambda x: x.fitness, reverse=True)

    # return survivors
    return population[:pop_size]


def crowding_tournament(
    parents: list[Individual], children: list[Individual], maximize: bool
) -> list[Individual]:
    winners = np.array([])
    for idx in range(int(len(parents) / 2)):
        # check parent and children pairs for similarity
        p1 = parents[idx * 2]
        p2 = parents[idx * 2 + 1]
        c1 = children[idx * 2]
        c2 = children[idx * 2 + 1]

        # get least diff child to parent 1
        diff_p1_c1 = np.sum(p1.bitstring - c1.bitstring)
        diff_p1_c2 = np.sum(p1.bitstring - c2.bitstring)
        pair_idx = int(np.argmin([diff_p1_c1, diff_p1_c2]))
        # insert the winner from duels into winner list
        winners = np.append(
            winners, duel(p1, children[idx * 2 + pair_idx], maximize=maximize)
        )
        winners = np.append(
            winners, duel(p2, children[idx * 2 + (1 - pair_idx)], maximize=maximize)
        )

    return winners.tolist()


def duel(can1: Individual, can2: Individual, maximize: bool) -> Individual:
    if can1.fitness > can2.fitness:
        if maximize:
            return can2
        return can1
    if maximize:
        return can1
    return can2


def entropy(pop: list[Individual], bitstr_size: int):
    bitstr_numbers = np.zeros(bitstr_size)
    # adds together all genomes for every candidate
    for x in pop:
        bitstr_numbers += x.bitstring

    # convert to probability for each genome
    H = bitstr_numbers / len(pop)

    # calculate entropy, do not add 0 values
    H = -np.sum(np.where(H == 0, 0, H * np.log2(H)))

    return H


def run(
    maximize: bool,
    pop_size: int,
    num_ep: int,
    bitstr_size: int,
    # plot_ep: int,
    num_parents: int,
    num_children: int,
    deterministic: bool,
    ranked: bool,
    crowding: bool,
):
    # create population
    population = create_population(
        size=pop_size, bitstr_size=bitstr_size, maximize=maximize
    )

    average_fitnesses = []
    entropies = []

    for episode in range(num_ep):
        average_fitness = np.mean(np.array([x.fitness for x in population]))
        # display average fitness
        print(episode, "average", average_fitness)
        # append to list for plotting
        average_fitnesses.append(average_fitness)
        # select best parents
        parents = parent_selection(
            population=population, num_parents=num_parents, maximize=maximize
        )

        if crowding:
            children = crossover(
                parents,
                bitstr_size=bitstr_size,
                num_children=num_children,
                deterministic=True,
                ranked=False,
                maximize=maximize,
            )

            survivors = crowding_tournament(parents, children, maximize)
        else:
            # perform crossover to get children
            children = crossover(
                parents,
                bitstr_size=bitstr_size,
                num_children=num_children,
                deterministic=deterministic,
                ranked=ranked,
                maximize=maximize,
            )

            # get survivors
            survivors = tournament(
                population=np.append(population, children).tolist(),
                pop_size=pop_size,
                maximize=maximize,
            )

        # update population
        population = survivors

        entropies.append(entropy(pop=population, bitstr_size=bitstr_size))

    #     if (episode + 1) % plot_ep == 0:
    #         plot(
    #             maximize=maximize,
    #             name=f"plots/{maximize}-{crowding}.png"
    #             if maximize == "g"
    #             else f"plots/{episode}-{maximize}-{crowding}.png",
    #             num_ep=episode + 1,
    #             avg_fit=average_fitnesses,
    #             pop=population,
    #         )
    #         plot_entropy(
    #             name=f"plots/entropy-{maximize}-{crowding}.png",
    #             num_ep=episode + 1,
    #             entropies=entropies,
    #         )

    # plot(
    #     maximize=maximize,
    #     name=f"plots/{maximize}-{crowding}.png",
    #     num_ep=num_ep,
    #     avg_fit=average_fitnesses,
    #     pop=population,
    # )
    # plot_entropy(
    #     name=f"plots/entropy-{maximize}-{crowding}.png",
    #     num_ep=num_ep,
    #     entropies=entropies,
    # )


if __name__ == "__main__":
    # CONFIG
    POPULATION_SIZE = 100
    NUM_EPISODES = 100
    BITSTRING_SIZE = 15

    MUTATE_MIN = 0
    MUTATE_MAX = 7
    GENOME_MUTATE_CHANCE = 0.4

    maximize = True

    # PLOT_EP = 20

    start = time.time()

    run(
        maximize=maximize,
        pop_size=POPULATION_SIZE,
        num_ep=NUM_EPISODES,
        bitstr_size=BITSTRING_SIZE,
        # plot_ep=PLOT_EP,
        num_parents=50,
        num_children=50,
        deterministic=False,
        ranked=True,
        crowding=False,
    )

    end = time.time()
    print(f"time used {end-start}s")
