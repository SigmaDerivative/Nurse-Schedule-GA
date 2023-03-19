import multiprocessing as mp

from ga import train_population, GeneticAlgorithm, EpochConfig, TrainConfig


class MultiProcessor:
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.ga_queue = mp.Queue(maxsize=num_workers)

    def start_processes(self, train_config: TrainConfig, epoch_config: EpochConfig):
        my_processes = []
        for i in range(self.num_workers):
            p = mp.Process(
                target=mp_train_population,
                args=(train_config, epoch_config, self.ga_queue),
            )
            p.start()
            my_processes.append(p)

    # TODO Fix this
    def combine_results(self) -> GeneticAlgorithm:
        ga = GeneticAlgorithm(size=0)
        for i in range(self.num_workers):
            ga += self.ga_queue.get()
        return ga


def mp_train_population(
    train_config: TrainConfig, epoch_config: EpochConfig, queue: mp.Queue
) -> None:
    ga = train_population(train_config, epoch_config)
    queue.put(ga)


if __name__ == "__main__":
    mp_ = MultiProcessor(num_workers=4)
    train_config = TrainConfig(pop_size=100, num_epoch=1000, print_num=100)
    epoch_config = EpochConfig(
        num_parents=20,
        num_new_clustered_individuals=2,
        num_new_random_individuals=4,
    )
    mp_.start_processes(train_config, epoch_config)
    ga = mp_.combine_results()
    print(ga)
