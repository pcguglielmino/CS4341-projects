# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from enum import Enum
from colorama import Fore, Back
from Q_WEIGHTS import Q_WEIGHTS


class Actions(Enum):
    UP = 0
    UP_RIGHT = 1
    RIGHT = 2
    RIGHT_DOWN = 3
    DOWN = 4
    DOWN_LEFT = 5
    LEFT = 6
    LEFT_UP = 7
    BOMB = 8
    STAY = 9


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


def calculate_reward(wrld, new_wrld):
    score_diff = new_wrld.scores["me"] - wrld.scores["me"]
    cost_of_living = 2
    return score_diff - cost_of_living


class TestCharacter(CharacterEntity):

    def __init__(self, name, avatar, x, y):
        CharacterEntity.__init__(self, name, avatar, x, y)
        self.learning_limit = 50
        self.learning_step = self.learning_limit
        self.is_learning = True
        self.discount = 0.9

    def do(self, wrld):
        # Your code here

        # List of functions to use to approximate the Q state
        lof = [self.distance_to_exit]

        low = self.get_next_worlds(wrld)    # list of worlds in (world, action.name) tuples

        loq = []    # list of q-values in (value, action.name, world) tuples

        # Calculate Q-values of possible world states
        for i in range(len(low)):
            loq.append((q_value(low[i][0][0], lof), low[i][1], low[i][0][0]))

        # print(loq)
        # print(max(loq))
        # print(locals()['self'])
        # print(getattr(locals()['self'], 'distance_to_exit')(wrld))

        action_tuple = max(loq)

        if self.is_learning:
            if self.learning_step > 0:

                q_s_a = action_tuple[0]

                alpha = self.learning_step/self.learning_limit

                r = calculate_reward(wrld, action_tuple[2])

                lonw = self.get_next_worlds(action_tuple[2])
                lonq = []

                for i in range(len(lonw)):
                    lonq.append((q_value(lonw[i][0][0], lof), lonw[i][1], lonw[i][0][0]))

                q_s_prime_a_prime = max(lonq)[0]

                delta = r + self.discount*q_s_prime_a_prime - q_s_a

                for i in Q_WEIGHTS:
                    Q_WEIGHTS[i] = Q_WEIGHTS[i] + alpha*delta*getattr(locals()['self'], i)(wrld)

                f = open("../Q_WEIGHTS.py", "w")

                f.write("Q_WEIGHTS = " + str(Q_WEIGHTS))

                f.close()

                self.learning_step -= 1
            else:
                exit("Done learning this episode")

        # Take the best action
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

    def distance_to_exit(self, wrld):
        # Distance to exit
        me = wrld.me(self)
        x = me.x
        y = me.y
        x_d = wrld.exitcell[0]
        y_d = wrld.exitcell[1]

        # Normalization factor
        norm = (wrld.width()**2 + wrld.height()**2)**.5

        return (((x-x_d)**2 + (y-y_d)**2)**.5)/norm

    def get_next_worlds(self, wrld):

        list_of_worlds = []

        me = wrld.me(self)

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
            elif member == Actions.BOMB:
                me.place_bomb()
            elif member == Actions.STAY:
                me.move(0, 0)
            list_of_worlds.append((wrld.next(), name))

        return list_of_worlds
