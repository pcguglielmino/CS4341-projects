# This is necessary to find the main code
import sys

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from enum import Enum
from colorama import Fore, Back
from Q_WEIGHTS import Q_WEIGHTS
from random import seed, random, choice
import math


class Actions(Enum):
    UP = 0
    UP_RIGHT = 1
    RIGHT = 2
    RIGHT_DOWN = 3
    DOWN = 4
    DOWN_LEFT = 5
    LEFT = 6
    LEFT_UP = 7
    # BOMB = 8
    STAY = 9


# This function iterates through the Q_table containing the function names and the weights of those functions,
# and calculates the Q-value of the given world using those functions
def q_value(new_wrld, lof):
    q = 0
    for f in lof:
        w = 0
        if Q_WEIGHTS.__contains__(f.__name__):
            w = Q_WEIGHTS[f.__name__]
        else:
            Q_WEIGHTS[f.__name__] = 0
        q += w * f(new_wrld)
    return q


# This function calculates the reward of going from one world to a new world
def calculate_reward(wrld, new_wrld):
    score_diff = new_wrld.scores["me"] - wrld.scores["me"]
    cost_of_living = 2
    return score_diff - cost_of_living


class TestCharacter(CharacterEntity):

    def __init__(self, name, avatar, x, y):
        CharacterEntity.__init__(self, name, avatar, x, y)
        self.learning_limit = 5000
        self.learning_step = self.learning_limit
        self.is_learning = False
        self.discount = 0.9
        self.epsilon_start = 0.95
        self.epsilon = self.epsilon_start
        self.epsilon_rate = 2  # How fast should epsilon decrease
        seed(2)

    def do(self, wrld):
        # Your code here
        me = wrld.me(self)

        print(self.get_next_move(me, wrld))

        # List of functions to use to approximate the Q state (add to this as more functions are implemented)
        lof = [self.distance_to_exit, self.is_wall]

        low = self.get_next_worlds(wrld)  # list of worlds in (world, action.name) tuples

        loq = []  # list of q-values in (value, action.name, world) tuples

        # Calculate Q-values of possible world states
        for i in range(len(low)):
            loq.append((q_value(low[i][0][0], lof), low[i][1], low[i][0][0]))

        # This tuple defines the action that the agent will take (value, action.name, world)
        action_tuple = self.select_action(loq)

        self.epsilon = self.epsilon_start * math.exp(-(1/(self.learning_limit*self.epsilon_rate))*(5000 - wrld.time))

        # If the agent is learning, this step will do the approximate Q-learning weight adjustment based on the function
        # learned in class. The agent will stop learning based on the learning_limit set in __init__()
        if self.is_learning:
            if self.learning_step > 0:

                q_s_a = action_tuple[0]
                alpha = self.learning_step / self.learning_limit
                r = calculate_reward(wrld, action_tuple[2])

                lonw = self.get_next_worlds(action_tuple[2])
                lonq = []

                for i in range(len(lonw)):
                    lonq.append((q_value(lonw[i][0][0], lof), lonw[i][1], lonw[i][0][0]))

                q_s_prime_a_prime = max(lonq)[0]
                delta = r + self.discount * q_s_prime_a_prime - q_s_a

                for i in Q_WEIGHTS:
                    Q_WEIGHTS[i] = Q_WEIGHTS[i] + alpha * delta * getattr(locals()['self'], i)(wrld)

                f = open("../Q_WEIGHTS.py", "w")
                f.write("Q_WEIGHTS = " + str(Q_WEIGHTS))
                f.close()

                self.learning_step -= 1
            else:
                exit("Done learning this episode")

        # Take the best action, based on the name of the action from the action_tuple
        self.move(0, 0)

        action = action_tuple[1]

        if action == 'UP':
            self.move(0, -1)
        elif action == 'UP_RIGHT':
            self.move(1, -1)
        elif action == 'RIGHT':
            self.move(1, 0)
        elif action == 'RIGHT_DOWN':
            self.move(1, 1)
        elif action == 'DOWN':
            self.move(0, 1)
        elif action == 'DOWN_LEFT':
            self.move(-1, 1)
        elif action == 'LEFT':
            self.move(-1, 0)
        elif action == 'LEFT_UP':
            self.move(-1, -1)
        elif action == 'BOMB':
            self.place_bomb()
        elif action == 'STAY':
            self.move(0, 0)

    # This function calculates the distance to the exit
    def distance_to_exit(self, wrld):
        me = wrld.me(self)
        if me is not None:
            x = me.x
            y = me.y
            x_d = wrld.exitcell[0]
            y_d = wrld.exitcell[1]

            # Normalization factor
            norm = (wrld.width() ** 2 + wrld.height() ** 2) ** .5

            return (((x - x_d) ** 2 + (y - y_d) ** 2) ** .5) / norm
        else:
            return 0

    def is_wall(self, wrld):
        # TODO: add a check for if there is a breakable wall next to you (one for each direction?)
        me = wrld.me(self)
        if me is not None:
            x = me.x
            y = me.y
            loc = self.get_adjacent(wrld)
            for cell in loc:
                if wrld.wall_at(cell[0], cell[1]):
                    return 1
            else:
                return 0
        else:
            return 0

    def is_bomb(self, wrld):
        # TODO: add a check for if there is a bomb next to you
        return 0

    def is_explosion(self, wrld):
        # TODO: add a check for if there is an explosion next to you
        return 0

    def is_monster(self, wrld):
        # TODO: add a check for if there is a monster next to you
        return 0

    # This function selects an action from the list of q_values
    def select_action(self, loq):
        if self.is_learning:
            val = random()
            if val < self.epsilon:
                return choice(loq)
            else:
                return max(loq)
        else:
            return max(loq)

    def get_next_worlds(self, wrld):
        list_of_worlds = []
        me = wrld.me(self)

        if me is not None:
            for name, member in Actions.__members__.items():
                if member == Actions.UP:
                    me.move(0, -1)
                elif member == Actions.UP_RIGHT:
                    me.move(1, -1)
                elif member == Actions.RIGHT:
                    me.move(1, 0)
                elif member == Actions.RIGHT_DOWN:
                    me.move(1, 1)
                elif member == Actions.DOWN:
                    me.move(0, 1)
                elif member == Actions.DOWN_LEFT:
                    me.move(-1, 1)
                elif member == Actions.LEFT:
                    me.move(-1, 0)
                elif member == Actions.LEFT_UP:
                    me.move(-1, -1)
                # elif member == Actions.BOMB:
                #     me.place_bomb()
                elif member == Actions.STAY:
                    me.move(0, 0)
                list_of_worlds.append((wrld.next(), name))
        else:
            list_of_worlds.append((wrld.next(), "DEAD"))

        return list_of_worlds

    # Get a list of all adjacent cells (i.e. not out of bounds)
    def get_adjacent(self, wrld):
        me = wrld.me(self)
        x = me.x
        y = me.y

        loc = []

        for i in range(-1, 2):
            for j in range(-1, 2):
                if wrld.width() > (x + i) >= 0 and wrld.height() > (y + j) >= 0:
                    loc.append((x + i, y + j))

        return loc

    def get_next_move(self, wrld):
        me = wrld.me(self)

        list_of_worlds = []

        if me is not None:
            for name, member in Actions.__members__.items():
                if member == Actions.UP:
                    me.move(0, -1)
                elif member == Actions.UP_RIGHT:
                    me.move(1, -1)
                elif member == Actions.RIGHT:
                    me.move(1, 0)
                elif member == Actions.RIGHT_DOWN:
                    me.move(1, 1)
                elif member == Actions.DOWN:
                    me.move(0, 1)
                elif member == Actions.DOWN_LEFT:
                    me.move(-1, 1)
                elif member == Actions.LEFT:
                    me.move(-1, 0)
                elif member == Actions.LEFT_UP:
                    me.move(-1, -1)
                list_of_worlds.append((wrld.next(), name))

        return list_of_worlds

    def heuristic(self, wrld):
        me = wrld.me(self)

        if me is not None:
            x = me.x
            y = me.y
            x_d = wrld.exitcell[0]
            y_d = wrld.exitcell[1]

            # Normalization factor
            norm = (wrld.width() ** 2 + wrld.height() ** 2) ** .5

            return (((x - x_d) ** 2 + (y - y_d) ** 2) ** .5) / norm
        else:
            return 0

    # def reconstruct_path(self, came_from, current):
    #     path = []
    #

    # A star algorithm based off the one on wikipedia
    def a_star(self, wrld):
        worlds = self.get_next_move(wrld)

        nodes_to_explore = []
        for w in worlds:
            nodes_to_explore.append((self.heuristic(w[0]) + 1, w[1]))

        came_from = {}

        g = 0
        g_score = {}

        h = self.heuristic(wrld)

        f = 0
        f_score = {}

        g_score[(wrld, "STAY")] = g
        f_score[(wrld, "STAY")] = g + h

        while len(nodes_to_explore) > 0:
            current = min(nodes_to_explore)

            me = current[1].me(self)

            # Made it to the exit (or died)
            if me is None:
                return self.reconstruct_path()

            nodes_to_explore.remove(current)

            new_worlds = self.get_next_move(current[1])

            for world in new_worlds:
                temp_g_score = g_score[current] + 1
                if temp_g_score < g_score

