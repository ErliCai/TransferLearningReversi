import numpy as np
from cnn import ReversiCNN
from players.player import Player
from copy import deepcopy


class RLPlayer(Player):
    """
    Deep RL Player (currently implemented like Sarsa with CNN
    as a function approximator)
    """

    def __init__(self, size, epsilon=0.01, seed=None, buffer_size=1000,
                 buffer_update_size=10, n=10, cnn_layers=[3, 4],
                 kernel=3, ff_layers=[10], lr=0.001, n_approx=1):
        """
        Create a player for a certain board size
        :param size:    Size of the board for which to create an agent
        :param epsilon: Exploration rate
        :param seed:    Random seed to use
        :param buffer_size: Size of the buffer to use with each function
                            approximator
        :param buffer_update_size: Number of learning updates to perform
                                   for each step
        :param n: as in TD(n)
        """
        super().__init__(size, epsilon, seed)
        # self.approximators are unique approximation functions
        # (several functions for different levels)
        self.approximators = [ReversiCNN(size, cnn_layers, kernel, ff_layers,lr)
                              for _ in range(n_approx)]
        self.buffers = [Buffer(buffer_size, size) for _ in self.approximators]
        # level_to_approx marks the game progression level (# pieces on board)
        # to the approximation function that must be used
        self.level_to_approx = {i: self.approximators[int(i / (size ** 2 + 1) * n_approx)]
                                for i in range(size ** 2 + 1)}
        self.level_to_buffer = {i: self.buffers[int(i / (size ** 2 + 1) * n_approx)]
                                for i in range(size ** 2 + 1)}
        self.buffer_update_size = buffer_update_size
        self.n = n

    @staticmethod
    def from_params(size, seed, params):
        params = deepcopy(params)
        return RLPlayer(size, epsilon=params["epsilon"], seed=seed,
                        buffer_size=params["buffer_size"],
                        buffer_update_size=params["buffer_update_size"],
                        n=params["n"], cnn_layers=params["cnn_layers"],
                        kernel=params["kernel"], ff_layers=params["ff_layers"],
                        lr=params["lr"], n_approx=params["n_approx"])

    def estimate(self, states):
        """
        Given a set of states which all have have the same number of pieces
        on the board, return estimated values for those states.
        :param states:
        :return:
        """
        cnn_input_vector = np.array([state.numpy() for state in states])
        level = states[0].pieces_on_board()
        return self.level_to_approx[level].evaluate(cnn_input_vector)

    def train(self, history):
        for i in range(0, len(history) - 2, 2):
            before = history[i]
            after = history[i + 2]
            for offset in range(self.n, 0, -2):
                if i + offset < len(history):
                    after = history[i + offset]
                    break

            level_after = after.pieces_on_board()
            level_before = before.pieces_on_board()
            if after.is_terminal():
                score = after.get_score()
                after = int(score[0] > score[1])
            else:
                to_estimate = after.numpy()
                after = self.level_to_approx[level_after].evaluate([to_estimate])
            before = before.numpy()

            self.level_to_buffer[level_before].add([before], [after])
            x, y = self.level_to_buffer[level_before].select(self.buffer_update_size)
            self.level_to_approx[level_before].train_on_batch(x, y)

    def save(self, dir):
        for i in range(len(self.approximators)):
            self.approximators[i].save(dir + "/" + str(i) + ".model")

    def load(self, dir):
        for i in range(len(self.approximators)):
            self.approximators[i].load(dir + "/" + str(i) + ".model")


class Buffer:
    """
    For buffering reversi states
    """

    def __init__(self, buffer_size, board_size):
        self.y = np.zeros(buffer_size)
        self.x = np.zeros(shape=(buffer_size, 2, board_size, board_size))
        self.buffer_total = 0
        self.next_in_buffer = 0

    def add(self, x, y):
        if len(x) + self.next_in_buffer > len(self.x):
            diff = len(x) + self.next_in_buffer - len(self.x)
            self.y[self.next_in_buffer:] = y[:-diff]
            self.y[0:diff] = y[-diff:]
            self.x[self.next_in_buffer:] = x[:-diff]
            self.x[0:diff] = x[-diff:]
            self.next_in_buffer = diff
        elif len(x) + self.next_in_buffer == len(self.x):
            self.y[self.next_in_buffer:] = y
            self.x[self.next_in_buffer:] = x
            self.next_in_buffer = 0
        else:
            self.y[self.next_in_buffer: self.next_in_buffer + len(y)] = y
            self.x[self.next_in_buffer: self.next_in_buffer + len(x)] = x
            self.next_in_buffer += len(x)
        self.buffer_total += len(x)

    def select(self, n):
        idx = np.random.randint(0, min(self.buffer_total, len(self.x)), n)
        return self.x[idx], self.y[idx]