import time

import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

import problem
from ga import GeneticAlgorithm, EpochConfig
from mutations import repair_random, repair_greedy
from utils import solution_to_list
from initializations import generate_random_genome
from mutations import destroy_violations


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    # gen = generate_random_genome()
    # problem.problem.visualize_solution(gen)
    # print(f"solution {solution_to_list(gen)}")
    # gen_ = destroy_violations(gen)
    # problem.problem.visualize_solution(gen_)
    # print(f"solution {solution_to_list(gen_)}")

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
        penalize_invalid=False,
    )

    for i in range(cfg.num_epochs):
        if i >= cfg.epoch.un_penalized_epochs:
            epoch_config.penalize_invalid = True
        ga.epoch(epoch_config)

        if i % cfg.print_num == 0:
            ga.sort_population_()
            print(i, ga.fitness[0][0])

    ga.sort_population_()
    best_fitness = ga.fitness[0][0]
    print(f"fitness {best_fitness}")
    print(f"solution {solution_to_list(ga.genomes[0])}")
    print(f"valid {ga.valids[0]}")
    problem.problem.visualize_solution(ga.genomes[0])

    return best_fitness


if __name__ == "__main__":

    # import cProfile, pstats

    # profiler = cProfile.Profile()
    # profiler.enable()

    main()

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats("profile/main.prof")
