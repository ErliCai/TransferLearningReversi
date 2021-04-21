import numpy as np
from players.player import Player


class TileWeightPlayer(Player):
    """
    A baseline player that evaluates a state by assigning different weights to
    different tiles and summing up the result
    """

    # default weights for certain board sizes:

    WEIGHTS = {6: [[100, -10, 5, 5, -10, 100],
                   [-10, -20, 2, 2, -20, -10],
                   [5, 2, 1, 1, 2, 5],
                   [5, 2, 1, 1, 2, 5],
                   [-10, -20, 2, 2, -20, -10],
                   [100, -10, 5, 5, -10, 100]],
               8: [[100, -20, 10, 5, 5, 10, -20, 100],
                   [-20, -50, -2, -2, -2, -2, -50, -20],
                   [10, -2, -1, -1, -1, -1, -2, 10],
                   [5, -2, -1, -1, -1, -1, -2, 5],
                   [5, -2, -1, -1, -1, -1, -2, 5],
                   [10, -2, -1, -1, -1, -1, -2, 10],
                   [-20, -50, -2, -2, -2, -2, -50, -20],
                   [100, -20, 10, 5, 5, 10, -20, 100]]}

    def __init__(self, size, weights=None, seed=None, minimax_depth=0):
        super().__init__(size, 0, seed, False, minimax_depth)
        if weights:
            self._weights = np.array(weights)
        elif size in TileWeightPlayer.WEIGHTS:
            self._weights = np.array(TileWeightPlayer.WEIGHTS[size])

    def estimate(self, states):
        return np.array([np.sum(s._board * self._weights) for s in states])