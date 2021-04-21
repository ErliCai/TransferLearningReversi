from players.weight_player import TileWeightPlayer
from players.minmoves_player import MinMovesPlayer
from task import *
from curriculum import *
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
from reversi import ReversiState
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import sys

TEST_N_GAMES = 500  # number of games to play when testing
# various kinds of tasks
TARGET = OpponentTask(opponents={6: TileWeightPlayer(6),
                                 8: TileWeightPlayer(8)},
                      name="target")
RANDOM = OpponentTask(opponents={6: TileWeightPlayer(6, [[0] * 6] * 6),
                                 8: TileWeightPlayer(8, [[0] * 8] * 8)},
                      name="random")
MOVE_BASED = OpponentTask(opponents={6: MinMovesPlayer(6, seed=2),
                                     8: MinMovesPlayer(6, seed=2)},
                          name="move-based")
OPPONENT_BASED = [SelfPlayTask(), RANDOM, MOVE_BASED]
START_BASED = [StartingPositionTask(2/3, 1), StartingPositionTask(1/3, 2),
               StartingPositionTask(0, 3)]
EXPLORATION_BASED = [ExplorationRateTask(0.01), ExplorationRateTask(0.1)]


def play(task, n_games, train=True, keep_results=False,
         seed=None, win_rate_cutoff=10, win_rate_frame=500):
    """
    Play n_games games between two players. For each game, randomly decide
    who goes first
    :param task             a task to train on
    :param n_games:         total number of games to play
    :param train:           whether to train the players after each batch.
    :param keep_results:    whether to preserve the scores for each game
    :param seed:            random seed to use to decide who plays first in
                            each game
    :param win_rate_cutoff: the cutoff at which to stop training
    :param win_rate_frame:  frame for which to calculate the winning rate
    :return:
    """
    results = []
    generator = np.random.default_rng(seed)
    bar = tqdm(range(n_games))
    for i in bar:
        players, starting_state = task.get_new_game()
        init_turn = 0 if generator.random() > 0.5 else 1
        turn = init_turn
        history = [starting_state]
        while (len(history) < 3) or (history[-1] != history[-3]):
            next_state = players[turn].act(history[-1])
            history.append(next_state)
            turn = (turn + 1) % 2
        score = history[-1].get_score()

        results.append([score[turn], score[(turn + 1) % 2]])
        if not keep_results:
            results = results[-win_rate_frame:]

        if train:
            history = np.array(history[:-1])
            players[init_turn].train(history[1:, ])
            players[(init_turn + 1) % 2].train(history[2:, ])

        if i != 0 and i % win_rate_frame == 0:
            frame = min(len(results), win_rate_frame)
            win_rate = sum([x[0] > x[1] for x in results[-frame:]])/frame
            bar.set_description("Win_rate: " + str(win_rate))
            if win_rate > win_rate_cutoff:
                break

    if not keep_results:
        return []
    return np.array(results)


def try_to_load(filename):
    try:
        with open(filename) as file:
            return file.readlines()
    except:
        return None


def train_test_save(params, task, size, n_games, dir, agent=None,
                    test_task=None, seed=0):
    """
    Train an agent on a given task for a given number of games on a given board
    size, then set exploration to False and test the agent on the given task.
    Record training history and the final winning rate in the given directory.
    If the agent with given parameters was already trained, retreive and
    return the result
    :param params:  the parameters with which to initialize the agent
    :param task:    the task to train\test on
    :param size:    size of the board
    :param n_games: number of games to train for
    :param dir:     directory to save the data to
    :param test_task: task on which to test. If none, use the first task
    :param agent:   if not None, use the given agent instead of creating new one
    :param seed:    random seed to use. For testing, use seed + 1
    :return:        the test winning rate
    """
    params = deepcopy(params)
    key_string = "|".join([key + ":" + str(params[key])
                           for key in sorted(params.keys())])
    full_dir = dir + "/" + str(size) + "/" + key_string
    loaded = try_to_load(full_dir + "/stats.txt")
    if loaded:
        return float(loaded[0].strip("\n"))
    if not agent:
        agent = RLPlayer.from_params(size, seed, params)
    task.init(agent, ReversiState(size))
    train_result = play(task, n_games, train=True, keep_results=True, seed=seed)
    agent.set_exploration(False)
    if not test_task:
        test_task = task
    else:
        test_task.init(agent, ReversiState(size))
    test_results = play(test_task, TEST_N_GAMES, train=False, keep_results=True,
                        seed=seed + 1)
    win_rate = np.sum(test_results[:, 0] > test_results[:, 1]) / \
               len(test_results)
    Path(full_dir).mkdir(parents=True, exist_ok=True)
    with open(full_dir + "/stats.txt", "w") as file:
        results = [str(win_rate)] + [str(result[0]) + "\t" + str(result[1])
                                     for result in train_result]
        file.writelines("\n".join(results))
    agent.save(full_dir)
    agent.set_exploration(True)
    return win_rate


def hyperparameter_search(task, size, n_games, dir, params, set_params,
                          best_params=None, seed=0, trace=True):
    """
    Perform greedy hyper-parameter search
    :param task:    the task on which to perform the search
    :param size:    the size of the board to perform the search for
    :param n_games: how long to continue training for
    :param dir:     the directory in which to store and record all the results
    :param params:  a list of parameters over which to perform the search
                    params[i][0] = parameter_name, params[i][1] - list of values
    :param set_params: parameters over which the search does not have to be
                       performed (as a dictionary)
    :param best_params: best parameters found so far
    :param seed:    random seed used by the agent and for training\testing loops
    :return:
    """
    if len(params) == 0:
        return train_test_save(set_params, task, size, n_games, dir, seed)
    new_set_params = deepcopy(set_params)
    for setting in params[1:]:
        new_set_params[setting[0]] = setting[1][0]
    best_result = 0
    param_name = params[0][0]
    if trace:
        print("Selecting best " + param_name)
    for param in params[0][1]:
        new_set_params[param_name] = param
        if trace:
            print("\tTrying " + str(param))
        new_result = hyperparameter_search(task, size, n_games, dir, [],
                                           new_set_params, trace=trace)
        if trace:
            print("\tGot " + str(new_result))
        if new_result >= best_result:
            best_params[param_name] = param
            best_result = new_result
    if trace:
        print("\tChoosing " + str(best_params[param_name]))
    set_params[param_name] = best_params[param_name]

    return hyperparameter_search(task, size, n_games, dir, params[1:],
                                 set_params, best_params, trace=trace)


def curriculum_tests(params, size, tasks, final_task, tasks_per_c,
                     games_per_task, n, dir, overwrite=False):
    """
    Conduct a slew of curriculum tests
    :param params: parameters with which to initialize every agent
    :param size:   size of the board to conduct the tests for
    :param tasks:  the tasks with which to generate the curricula
    :param final_task : task on which to train after all curricula
    :param tasks_per_c: tasks per curriculum
    :param games_per_task: number of games to play for each task
    :param n:      number of curricula to genenrate
    :param dir:    directory to save all the data to
    :param overwrite: if False, attempt to load a file with curricula
                      description before generating a new one
    :return:
    """
    meta_file = dir + "/meta_" + str(n) + "_" + str(tasks_per_c) + ".txt"
    try:
        curricula = Curriculum.load(meta_file, tasks)
    except:
        curricula = Curriculum.generate(tasks, n, tasks_per_c, final_task, 0)
        Curriculum.save(meta_file, curricula)
    curricula.insert(0, Curriculum([final_task] * (tasks_per_c + 1)))

    win_rates = []

    for i, curriculum in enumerate(curricula):
        print("\n\nTraining an agent on curriculum # %d" % i)
        win_rate = curriculum.train(params, size, games_per_task,
                                    final_task, dir=dir + "/" + str(i),
                                    seed=i)
        print("Achieved win rate of %f" % win_rate)
        win_rates.append(win_rate)

    return win_rates


def make_plot(xname, yname, values, legends, filename, start,
              symbols=("c*", "yo", "k+", "g^", "b*")):
    """
    Create a plot with an arbitrary umber of series
    :param xname:       x-axis title
    :param yname:       y-axis title
    :param values:      the series to plot (list of lists of values)
    :param symbols:     for each series in values, a symbol
    :param legends:     for each series in values, a description
    :param filename:    name of the file to save everything to
    :return:
    """
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    plot = fig.add_subplot()
    plot.set_xlabel(xname, fontdict={"size": 18})
    plot.set_ylabel(yname, fontdict={"size": 18})
    colors=[x[0] for x in symbols]
    data = [plot.scatter(np.arange(len(values[i])) + start[i], values[i],
                         c=colors[i]) for i in range(len(values))]
    plot.legend(data, legends, fontsize=16)
    fig.savefig(filename)


def load_curriculum_stats(filenames):
    result = []
    for filename in filenames:
        with open(filename) as file:
            lines = file.readlines()
        data = [[int(x) for x in line.strip("\n").split("\t")]
                for line in lines[1:]]
        data = [x[0] > x[1] for x in data]
        result = result + data
    return np.array(result, dtype=int)


def plot_training(filesets, legends, outname, window=1000, start=None):
    if not start:
        start = np.zeros(len(filesets))
    series = []
    for files in filesets:
        data = load_curriculum_stats(files)
        series.append([sum(data[max(i-window//2, 0):
                                min(i+window//2, len(data))])/
                       (min(i+window//2, len(data)) - max(i-window//2, 0))
                       for i in range(len(data))])
    make_plot("Games played", "Winning rate (window=1000)", series, legends,
              outname, start)


def do_hyper_parameter_tests(n_games=20000, trace=True):
    """
    Perform hyper-parameter search and return the best parameters both for the
    6 by 6 and the 8 by 8 board
    :return:
    """
    best_params6 = {}
    if trace:
        print("Conducting hyper-parameter search for 6 by 6 Reversi:")
    hyperparameter_search(TARGET, 6, seed=0, n_games=n_games,
                          dir="./results/hyper", best_params=best_params6,
                          params=[("n_approx", [1, 2, 4]),
                                  ("n", [20, 10, 2]),
                                  ("lr", [0.001, 0.01, 0.0001]),
                                  ("cnn_layers", [[3], [3, 6], [3, 6, 12]]),
                                  ("buffer_size", [3000, 1000]),
                                  ("epsilon", [0.01, 0.05])],
                          set_params={"buffer_update_size": 10, "kernel": 3,
                                      "ff_layers": [10]}, trace=trace)
    best_params6["buffer_update_size"] = 10
    best_params6["kernel"] = 3
    best_params6["ff_layers"] = [10]
    print("Best parameters for 6 by 6 Reversi: " + str(best_params6) + "\n\n")

    best_params8 = {}
    if trace:
        print("Conducting hyper-parameter search for 6 by 6 Reversi:")
    hyperparameter_search(TARGET, 8, seed=0, n_games=n_games,
                          dir="./results/hyper", best_params=best_params8,
                          params=[("n_approx", [1, 2]),
                                  ("n", [20, 10, 2]),
                                  ("lr", [0.001, 0.01, 0.0001]),
                                  ("kernel", [3, 4]),
                                  ("buffer_size", [3000, 1000]),
                                  ("ff_layers", [[10], [50, 10]]),
                                  ("cnn_layers", [[3], [3, 6], [3, 6, 12]]),
                                  ("epsilon", [0.01, 0.05])],
                          set_params={"buffer_update_size": 10}, trace=trace)
    best_params8["buffer_update_size"] = 10
    print("Best parameters for 8 by 8 Reversi: " + str(best_params8))

    return best_params6, best_params8


def do_long_tests(n_games=80000):
    """
    Runs both the 6x6 and 8x8 agents with optimal parameters on a large number of games.
    :return:
    """
    print("Training with best hyper-parameters for " + str(n_games) + " games:")
    print("Loading best hyper-parameters...")
    best_params6, best_params8 = do_hyper_parameter_tests(n_games=20000, trace=False)
    print("Training on 6x6 board...")
    winrate6 = train_test_save(best_params6, TARGET, 6, n_games, dir="./results/long")
    print("Training on 8x8 board...")
    winrate8 = train_test_save(best_params8, TARGET, 8, n_games, dir="./results/long")


def do_curriculum_tests(n_games=3000):
    """
    Performs all the curriculum tests and reproduce Figure 4.
    """
    print("Loading best hyper-parameters...")
    best_params6, best_params8 = do_hyper_parameter_tests(n_games=20000,
                                                          trace=False)
    tasks = [OPPONENT_BASED, EXPLORATION_BASED, START_BASED]
    final_task = TARGET.with_other_tasks([EXPLORATION_BASED[0]])
    print("Ranking curricula on 6x6 board...")
    rates6 = curriculum_tests(best_params6, 6, tasks=tasks,
                              final_task=final_task, tasks_per_c=2, n=20,
                              dir="./results/curriculum", overwrite=False,
                              games_per_task=n_games)

    # do curriculum learning experiments:
    rates8 = curriculum_tests(best_params8, 8, tasks=tasks,
                              final_task=final_task, tasks_per_c=2, n=20,
                              dir="./results/curriculum", overwrite=False,
                              games_per_task=n_games)
    rates6 = [round(x * 10) for x in rates6]  # ignore insignificant differences
    rates8 = [round(x * 10) for x in rates8]
    print("Print Kendall's tau:" + str(stats.kendalltau(rates6, rates8)))


def do_weight_transfer_tests(n_games=9000):
    """Do the weight transfer experiment and reproduce Figure 5"""
    print("Loading best hyper-parameters...")
    best_params6, best_params8 = do_hyper_parameter_tests(n_games=20000,
                                                          trace=False)
    best_params8["cnn_layers"] = [3]
    print("Training an agent on 8x8 board only for " + str(n_games) + " games")
    train_test_save(best_params8, TARGET, 8, n_games, dir="results/transfer/base")

    agent6 = RLPlayer.from_params(6, 0, best_params6)
    print("Training another agent on 6x6 board for " + str(int(n_games*5/18)) + " games")
    train_test_save(best_params6, TARGET, 6, int(n_games*5/18), dir="results/transfer/tran",
                    agent=agent6)
    print("Transferring weights...")
    state_dict6 = agent6.approximators[0].state_dict()
    agent8 = RLPlayer.from_params(8, 0, best_params8)
    state_dict8 = agent8.approximators[0].state_dict()
    state_dict8['convs.0.bias'] = state_dict6['convs.0.bias']
    state_dict8['convs.0.weight'] = state_dict6['convs.0.weight']
    state_dict8['linear.1.bias'] = state_dict6['linear.1.bias']
    state_dict8['linear.1.weight'] = state_dict6['linear.1.weight']
    agent8.approximators[0].load_state_dict(state_dict8)
    print("Training another agent on 8x8 board for " + str(
        int(n_games * 13 / 18)) + " gamesm initilizing with transferred weights")
    win_rate = train_test_save(best_params8, TARGET, 8, int(n_games*13/18),
                               dir="results/transfer/tran", agent=agent8)
    print("Win rate after transfer learning weights: " + str(win_rate))


def reproduce_figures():
    print("Reproducing Figure 3...")
    plot_training([["results/long/6/buffer_size:1000|buffer_update_size:10|cnn_layers:[3]|epsilon:0.01|ff_layers:[10]|kernel:3|lr:0.01|n:2|n_approx:1/stats.txt"],
                   ["results/long/8/buffer_size:3000|buffer_update_size:10|cnn_layers:[3, 6]|epsilon:0.01|ff_layers:[10]|kernel:3|lr:0.001|n:10|n_approx:1/stats.txt"]],
                  ["6x6", "8x8"], "Figure3.png", window=1000)

    print("Rerproducing Figure 5...")
    plot_training([["results/transfer/base/8/buffer_size:3000|buffer_update_size:10|cnn_layers:[3]|epsilon:0.01|ff_layers:[10]|kernel:3|lr:0.001|n:10|n_approx:1/stats.txt"],
                   ["results/transfer/tran/6/buffer_size:1000|buffer_update_size:10|cnn_layers:[3]|epsilon:0.01|ff_layers:[10]|kernel:3|lr:0.01|n:2|n_approx:1/stats.txt"],
                   ["results/transfer/tran/8/buffer_size:3000|buffer_update_size:10|cnn_layers:[3]|epsilon:0.01|ff_layers:[10]|kernel:3|lr:0.001|n:10|n_approx:1/stats.txt"]],
                  legends=["8x8 target", "6x6 target", "8x8 after 6x6"],
                  outname="Figure5.png", window=1000, start=[0, 0, 2500])

    print("Reproducing Figure 4...")
    plot_training([["results/curriculum/0/target_epsilon=0.01_0/8/buffer_size:3000|buffer_update_size:10|cnn_layers:[3, 6]|epsilon:0.01|ff_layers:[10]|kernel:3|lr:0.001|n:10|n_approx:1/stats.txt",
                    "results/curriculum/0/target_epsilon=0.01_1/8/buffer_size:3000|buffer_update_size:10|cnn_layers:[3, 6]|epsilon:0.01|ff_layers:[10]|kernel:3|lr:0.001|n:10|n_approx:1/stats.txt",
                    "results/curriculum/0/target_epsilon=0.01_2/8/buffer_size:3000|buffer_update_size:10|cnn_layers:[3, 6]|epsilon:0.01|ff_layers:[10]|kernel:3|lr:0.001|n:10|n_approx:1/stats.txt"],
                   ["results/curriculum/2/random_epsilon=0.1_start-at-depth=0.67_0/8/buffer_size:3000|buffer_update_size:10|cnn_layers:[3, 6]|epsilon:0.01|ff_layers:[10]|kernel:3|lr:0.001|n:10|n_approx:1/stats.txt"],
                   ["results/curriculum/2/self-play_epsilon=0.1_start-at-depth=0.33_1/8/buffer_size:3000|buffer_update_size:10|cnn_layers:[3, 6]|epsilon:0.01|ff_layers:[10]|kernel:3|lr:0.001|n:10|n_approx:1/stats.txt"],
                   ["results/curriculum/2/target_epsilon=0.01_2/8/buffer_size:3000|buffer_update_size:10|cnn_layers:[3, 6]|epsilon:0.01|ff_layers:[10]|kernel:3|lr:0.001|n:10|n_approx:1/stats.txt"]],
                  ["No curriculum, target task", "Curriculum, sub-task # 1",
                   "Curriculum, sub-task # 2", "Curriculum, target task"],
                  "Figure4.png", window=1000,
                  start=[0, 0, 3000, 6000])


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Reproduce the experiments from the report")
    p.add_argument("--hyper", dest="hyper", action="store_true",
                   help="Do hyper-parameter search (Sections 4.2, 5.1 in the report)")
    p.add_argument("--long", dest="long", action="store_true",
                   help="Train on many games with best hyper parameters (Sections 5.1)")
    p.add_argument("--curriculum", dest="curriculum", action="store_true",
                   help="Rank and compare curricula's performance for different board sizes (Section 5.2)")
    p.add_argument("--transfer", dest="transfer", action="store_true",
                   help="Do transfer weight experiment from 6x6 to 8x8 board (Section 5.3)")
    p.add_argument("--figures", dest="figures", action="store_true",
                   help="Reproduce the figures")
    args = p.parse_args(sys.argv[1:])

    if args.hyper:
       do_hyper_parameter_tests()
    if args.long:
        do_long_tests()
    if args.curriculum:
        do_curriculum_tests()
    if args.transfer:
        do_weight_transfer_tests()
    if args.figures:
        reproduce_figures()