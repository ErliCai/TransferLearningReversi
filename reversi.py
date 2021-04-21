import numpy as np
from copy import deepcopy


class ReversiState:
    """ Represents an (after)state in Reversi """

    def __init__(self, size):
        """
        Create a starting state
        :param size: Size of the board (integer)
        """
        self._size = size

        # Initialize the board with a 2d array
        board = [0] * size * size
        p = int((size/2 - 1) * size + size/2 - 1)
        board[p] = board[p + size + 1] = 1
        board[p + 1] = board[p + size] = -1
        self._board = np.array(board)
        self._board = self._board.reshape(size, size)

    def numpy(self):
        """
        Return the state as a 3 2d numpy array. For each tile,
            one represents unoccupied tiles
            one represents tiles occupied by the player whose turn it is
            one represents tiles occupied by the opposing player
        :return:
        """
        occupied1 = np.zeros((self._size, self._size) , dtype=object)
        occupied2 = np.zeros((self._size, self._size) , dtype=object)

        for i in range(self._size):
            for j in range(self._size):
                if self._board[i][j] == 1:
                    occupied1[i][j] = 1
                elif self._board[i][j] == -1:
                    occupied2[i][j] = 1

        return np.array([occupied1, occupied2], dtype=np.float)

    def afterstates(self):
        """
        Return a list of states reachable from self in one move.
        Note: if the current player cannot make a move, return the same state
        :return: List<ReversiState>
        """
        reachable_states = []
        directions = [[1,0], [-1,0], [0,1], [0,-1], [1,1], [1,-1], [-1,1], [-1,-1]]
        # Iterate on all the space and find those not occupied by any player yet
        # Then try all possible directions on those space
        for i in range(self._size):
            for j in range(self._size):
                if self._board[i, j] == 0:
                    updates = []
                    for d in directions:
                        i0 = i+d[0]
                        j0 = j+d[1]
                        possible_updates = []
                        while ( (0<=i0<self._size)
                            and (0<=j0<self._size)
                            and (self._board[i0, j0] == -1) ):
                            possible_updates.append([i0, j0])
                            i0 = i0 + d[0]
                            j0 = j0 + d[1]
                        if ( (0<=i0<self._size)
                            and (0<=j0<self._size)
                            and self._board[i0][j0] == 1):
                            updates += possible_updates
                    if updates:
                        nextstate = deepcopy(self)
                        nextstate._board[i][j] = 1
                        for k in updates:
                            nextstate._board[k[0], k[1]] = 1
                        reachable_states.append(-nextstate)

        if not reachable_states:
            return [-self]
        else:
            return reachable_states

    def get_score(self):
        """
        If this is a terminal state, return the scores of both players as a
        tuple (score_of_the_player_to_make_a_move, score_of_the_other_player)
        :return: a tuple of two integers
        """
        return np.sum(self._board == 1), np.sum(self._board == -1)

    def pieces_on_board(self):
        """ Return the number of pieces left on board """
        return np.sum(self._board != 0)

    def is_terminal(self):
        """
        Returns True iff nether player can make a move
        :return:
        """
        if self.pieces_on_board() == self._size ** 2:
            return True
        return self == self.afterstates()[0].afterstates()[0]

    def get_size(self):
        return self._size

    def __hash__(self):
        """
        Symmetric states hash to the same value. So do the states that are
        same expect that the color of all the pieces on the board AND that of
        the player making the move are flipped.
        """
        # TODO
        # I find it incredibly hard to include more than one symmetry in the
        # hash function. Still need to figure out how to do it

    def __eq__(self, other):
        """
        States that hash to the same value are considered equal
        :param other: a ReversiState
        :return:
        """
        for i in range(len(self._board)):
            for j in range(len(self._board[i])):
                if self._board[i][j] != other._board[i][j]:
                    return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __neg__(self):
        negative = deepcopy(self)
        negative._board = -negative._board
        return negative

    @staticmethod
    def tile_to_str(i):
        if i == 1:
            return "X"
        elif i == -1:
            return "O"
        return "."

    def __str__(self):
        return "\n".join(["".join([self.tile_to_str(x) for x in y]) for y in self._board])


if __name__ == '__main__':
    # To test functionality
    r = ReversiState(6)
    print(r._board[2,2] == 1)
    print(r.afterstates())
    r._board[4,1] = -1
    r._board[4,2] = 1
    print(r.afterstates())
    print(r.numpy())
