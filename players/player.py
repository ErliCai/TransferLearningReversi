import numpy as np


class Player:
    """ An abstract class to be extended by different Rerversi players"""

    def __init__(self, size, epsilon, seed=None, exploration=True,
                 minimax_depth=0):
        """
        Create a Reversi player for a certain board size
        :param size: size of the Reversi board
        :param epsilon: exploration rate
        :param seed: seed to use for the random number generator
        :param exploration: whether to turn on or off the exploration
        :param minimax_depth: the depth of the minimax approach
        """
        self._size = size
        self._epsilon = epsilon
        self._generator = np.random.default_rng(seed)
        self._exploration = exploration
        self._minimax_depth = minimax_depth

    def set_exploration(self, value):
        self._exploration = value

    def set_epsilon(self, value):
        self._epsilon = value

    def set_minimax_depth(self, value):
        self._minimax_depth = value

    def choose_to_explore(self):
        return self._exploration and self._generator.random() < self._epsilon

    def estimate(self, states):
        """
        For each state in states, return state value estimates
        The estimates should be returned with respect to the player making the
        move
        :param states:
        :return:
        """
        scores = [state.get_score() for state in states]
        return np.array([score[0] - score[1] for score in scores])

    def act(self, state):
        """
        Given a state, choose among all possible afterstates
        :param state:
        :return:
        """

        afterstates = state.afterstates()

        if self.choose_to_explore():  # explore
            action_id = self._generator.integers(0, len(afterstates))
            return afterstates[action_id]

        estimates = self.act_with_minimax(afterstates, self._minimax_depth,
                                          maximizing=False, best=None,
                                          initial=True)
        min = np.amin(estimates)
        argmin = np.argwhere(estimates == min).flatten()
        action_id = argmin[self._generator.integers(0, len(argmin))]
        return afterstates[action_id]

    def act_with_minimax(self, states, minmax_left, maximizing, best,
                         initial=False):
        # leaf node case
        if minmax_left == 0:
            estimates = self.estimate(states)
            if initial:
                return estimates
            return self.minmax(estimates, maximizing)

        # not a leaf node
        estimates = [self.act_with_minimax(states[0].afterstates(),
                                           minmax_left - 1,
                                           maximizing=not maximizing,
                                           best=None)]
        curr_best = estimates[0]
        for i in range(1, len(states)):
            if best is not None:
                if maximizing and curr_best >= best:
                    return curr_best + 1
                elif not maximizing and curr_best <= best:
                    return curr_best - 1
            estimates.append(self.act_with_minimax(states[i].afterstates(),
                                                   minmax_left-1,
                                                   maximizing=not maximizing,
                                                   best=curr_best))
            curr_best = self.minmax([curr_best, estimates[-1]], maximizing)

        if initial:
            return estimates
        return curr_best

    def train(self, history):
        pass

    def minmax(self, a, maximizing):
        if maximizing:
            return max(a)
        return min(a)
