from main import train_test_save
import numpy as np
from players.rl_player import RLPlayer
from reversi import ReversiState
from pathlib import Path


class Curriculum:

    def __init__(self, tasks):
        self.tasks = tasks

    def train(self, params, size, games_per_task, final_task, dir, seed=None):
        agent = RLPlayer.from_params(size=size, seed=seed, params=params)
        win_rate = 0
        for i, task in enumerate(self.tasks):
            task.init(agent, ReversiState(size))
            print("Training the agent on the " + task.get_name() + " task")
            win_rate = train_test_save(params, task, size, games_per_task,
                                       dir + "/" + task.get_name() + "_" + str(i),
                                       agent=agent, test_task=final_task,
                                       seed=i)
        return win_rate

    @staticmethod
    def generate(raw_tasks, n_to_generate, tasks_per_curriculum, last_task,
                 seed=None):
        generator = np.random.default_rng(seed)
        combined_tasks = Curriculum.combine_tasks(raw_tasks)
        combined_tasks = [x[0].with_other_tasks(x[1:]) for x in combined_tasks]
        curricula = []
        keys = set()
        while len(curricula) < n_to_generate:
            curriculum, key = Curriculum.generate_one(combined_tasks,
                                                      tasks_per_curriculum,
                                                      last_task, generator)
            if key not in keys:
                keys.add(key)
                curricula.append(curriculum)
        return curricula

    @staticmethod
    def generate_one(combined_tasks, tasks_per_curriculum, last_task,
                     generator):
        key = ""
        tasks = []
        for _ in range(tasks_per_curriculum):
            tasks.append(generator.choice(combined_tasks))
            key += tasks[-1].get_name()
        tasks.append(last_task)
        return Curriculum(tasks), key

    @staticmethod
    def combine_tasks(tasks):
        if len(tasks) == 1:
            return [[x] for x in tasks[0]]
        result = []
        for task1 in tasks[0]:
            for task2 in Curriculum.combine_tasks(tasks[1:]):
                result.append([task1] + task2)
        return result

    @staticmethod
    def load(filename, tasks):
        lines = open(filename).readlines()
        tasks = [item for sublist in tasks for item in sublist]
        task_dict = {item.get_name(): item for item in tasks}
        curricula = []
        for line in lines:
            if line == "\n":
                continue
            task_strings = line.strip("\n").split("\t")
            curr_tasks = []
            for string in task_strings:
                string = string.split("_")
                task = task_dict[string[0]]
                if len(string) > 0:
                    task = task.with_other_tasks([task_dict[s]
                                                  for s in string[1:]])
                curr_tasks.append(task)
            curricula.append(Curriculum(curr_tasks))
        return curricula

    @staticmethod
    def save(filename, curricula):
        Path("/".join(filename.split("/")[:-1])).mkdir(
            parents=True, exist_ok=True)
        with open(filename, "w") as file:
            for curriculum in curricula:
                string = "\t".join([task.get_name()
                                    for task in curriculum.tasks])
                file.write(string + "\n")
