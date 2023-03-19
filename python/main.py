import time

import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

import problem
from ga import GeneticAlgorithm, EpochConfig


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    print(cfg.size)
    ga = GeneticAlgorithm(size=cfg.size)

    epoch_config = EpochConfig(num_survivors=95, num_mutators=94)

    for i in tqdm(range(20)):
        ga.epoch(epoch_config)

        if i % 10 == 0:
            ga.sort_population_()
            print(i, ga.fitness[0][0])

    problem.problem.visualize_solution(ga.genomes[0])
    return np.min(ga.fitness)


if __name__ == "__main__":
    main()
