import time

import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

import problem
from ga import GeneticAlgorithm, EpochConfig
from mutations import repair_random, repair_greedy


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    ga = GeneticAlgorithm(size=cfg.size)

    repair_function = {"random": repair_random, "greedy": repair_greedy}

    epoch_config = EpochConfig(
        num_parents=cfg.epoch.num_parents,
        num_new_clustered_individuals=cfg.epoch.new_clustered,
        num_new_random_individuals=cfg.epoch.new_random,
        n_destroys=cfg.epoch.n_destroys,
        repair_function=repair_function[cfg.epoch.repair_function],
        n_children=cfg.epoch.n_children,
        mate_elite_prob_factor=cfg.epoch.mate_elite_prob_factor,
    )

    for i in range(cfg.num_epochs):
        ga.epoch(epoch_config)

        if i % cfg.print_num == 0:
            ga.sort_population_()
            print(i, ga.fitness[0][0])

    ga.sort_population_()
    best_fitness = ga.fitness[0][0]
    print(f"fitness {best_fitness}")
    problem.problem.visualize_solution(ga.genomes[0])

    return best_fitness


if __name__ == "__main__":
    main()
