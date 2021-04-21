import numpy as np
from players.player import Player


class MinMovesPlayer(Player):
    """
    A baseline player that tries to minimize the number of moves that the
    opponent can make
    """

    def __init__(self, size, seed=None, minimax_depth=0):
        super().__init__(size, 0, seed, False, minimax_depth)

    def estimate(self, states):
        return np.array([len(s.afterstates()) for s in states])
