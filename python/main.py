import time

import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

import problem
from ga import GeneticAlgorithm, EpochConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    ga = GeneticAlgorithm(size=cfg.size)

    epoch_config = EpochConfig(
        num_parents=cfg.epoch.num_parents,
        num_new_clustered_individuals=cfg.epoch.new_clustered,
        num_new_random_individuals=cfg.epoch.new_random,
    )

    for i in range(cfg.num_epochs):
        ga.epoch(epoch_config)

        if i % cfg.print_num == 0:
            ga.sort_population_()
            print(i, ga.fitness[0][0])

    ga.sort_population_()
    print(f"fitness {ga.fitness[0][0]}")
    problem.problem.visualize_solution(ga.genomes[0])


if __name__ == "__main__":
    main()
