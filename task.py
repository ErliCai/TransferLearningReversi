from copy import deepcopy
import numpy as np


class Task:

    def __init__(self, name):
        self.name = name
        self.other_tasks = None
        self.next_task = None

    def with_other_tasks(self, other_tasks):
        new_task = deepcopy(self)
        new_task.other_tasks = other_tasks
        return new_task

    def init(self, player, starting_state):
        self.player = player
        self.starting_state = starting_state

        if self.other_tasks:
            self.next_task = self.other_tasks[0].\
                with_other_tasks(self.other_tasks[1:])
            self.next_task.init(player, starting_state)

    def get_new_game(self):
        player1, player2, starting_state = self.setup_new_game(self.player,
                                                               None,
                                                               self.starting_state)
        if self.next_task:
            player1, player2, starting_state = self.setup_new_game(player1,
                                                                   player2,
                                                                   starting_state)
        return [player1, player2], starting_state

    def setup_new_game(self, player1, player2, starting_state):
        return player1, player2, starting_state

    def get_name(self):
        result = self.name
        for task in self.other_tasks:
            result += "_" + task.name
        return result


class OpponentTask(Task):

    def __init__(self, opponents, name):
        super(OpponentTask, self).__init__(name)
        self.opponents = opponents

    def init(self, player, starting_state):
        super(OpponentTask, self).init(player, starting_state)
        self.opponent = self.opponents[starting_state._size]

    def setup_new_game(self, player1, player2, starting_state):
        return player1, self.opponent, starting_state


class SelfPlayTask(Task):

    def __init__(self):
        super(SelfPlayTask, self).__init__("self-play")

    def setup_new_game(self, player1, player2, starting_state):
        return player1, player1, starting_state


class StartingPositionTask(Task):

    def __init__(self, depth=0.5, seed=None):
        super(StartingPositionTask, self).__init__(
            "start-at-depth=" + str(round(depth, 2)))
        self.depth = depth
        self.generator = np.random.default_rng(seed)

    def setup_new_game(self, player1, player2, starting_state):
        make_moves = int((starting_state._size - 4) * self.depth)
        for _ in range(make_moves):
            starting_state = self.generator.choice(starting_state.afterstates())
        return player1, player2, starting_state


class ExplorationRateTask(Task):

    def __init__(self, rate=0.5):
        super(ExplorationRateTask, self).__init__(
            "epsilon="+str(round(rate, 2)))
        self.rate = rate

    def setup_new_game(self, player1, player2, starting_state):
        player1.set_epsilon(self.rate)
        return player1, player2, starting_state
