# This is necessary to find the main code
import sys

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from enum import Enum
from colorama import Fore, Back
from Q_WEIGHTS import Q_WEIGHTS
from random import seed, random, choice
from queue import PriorityQueue
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
    BOMB = 8
    STAY = 9
    FOLLOW_A_STAR = 10
    RUN_AWAY_BOMB = 11
    RUN_AWAY_EXPLOSION = 12


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
    events = new_wrld[1]
    event_score = 0
    for e in events:
        if e.tpe == 4:
            event_score += 1000
        if e.tpe == 2 or e.tpe == 3 and e.character.name == 'me':
            event_score -= 100000
        if e.tpe == 0:
            event_score += 10
        if e.tpe == 1:
            event_score += 100

    score_diff = new_wrld[0].scores["me"] - wrld.scores["me"]
    cost_of_living = 2
    return score_diff - cost_of_living + event_score


class TestCharacter(CharacterEntity):

    def __init__(self, name, avatar, x, y):
        CharacterEntity.__init__(self, name, avatar, x, y)
        self.learning_limit = 100
        self.learning_step = self.learning_limit
        self.is_learning = False
        self.discount = 0.9
        self.epsilon_start = 0.95
        self.epsilon = self.epsilon_start
        self.epsilon_rate = 0.3  # How fast should epsilon decrease
        self.path = []
        self.path_search = 0
        self.danger_radius = 5
        seed(10)

    def do(self, wrld):
        # List of functions to use to approximate the Q state (add to this as more functions are implemented)

        self.path = self.get_a_star(wrld)
        self.path_search = 0

        lof = [self.path_length, self.nearby_monster, self.bomb_threat, self.explosion_threat, self.nearby_wall, self.is_there_monster]

        low = self.get_next_worlds(wrld)  # list of worlds in (world, action.name) tuples

        loq = []  # list of q-values in (value, action.name, world, event) tuples

        # Calculate Q-values of possible world states
        for i in range(len(low)):
            loq.append((q_value(low[i][0][0], lof), low[i][1], low[i][0][0], low[i][0][1]))

        # This tuple defines the action that the agent will take (value, action.name, world, events)
        action_tuple = self.select_action(loq)

        self.epsilon = self.epsilon_start * math.exp(-(1/(self.learning_limit*self.epsilon_rate))*(5000 - wrld.time))

        # If the agent is learning, this step will do the approximate Q-learning weight adjustment based on the function
        # learned in class. The agent will stop learning based on the learning_limit set in __init__()
        if self.is_learning:
            if self.learning_step > 0:

                q_s_a = action_tuple[0]
                alpha = self.learning_step / self.learning_limit
                r = calculate_reward(wrld, (action_tuple[2], action_tuple[3]))

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

        if action == 'FOLLOW_A_STAR':
            if self.path is None:
                self.move(0, 0)
            else:
                action = self.coord_to_action(self.path.pop(0))

        if action == 'RUN_AWAY_BOMB':
            action = self.coord_to_action(self.run_away_bomb(wrld))

        if action == 'RUN_AWAY_EXPLOSION':
            action = self.coord_to_action(self.run_away_explosion(wrld))

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
            return 1

    def path_length(self, wrld):
        path = self.get_a_star(wrld)
        if path is not None:
            return 1 - (len(path) / (wrld.height() ** 2 + wrld.width() ** 2) ** 0.5)
        else:
            return 1

    def get_a_star(self, wrld):
        me = wrld.me(self)
        if me is not None:
            x = me.x
            y = me.y
            x_d = wrld.exitcell[0]
            y_d = wrld.exitcell[1]

            return a_star(wrld, (x, y), (x_d, y_d))

    def nearby_monster(self, wrld):
        me = wrld.me(self)
        if me is not None:
            x = me.x
            y = me.y
            width = wrld.width()
            heigth = wrld.height()
            if x - 1 >= 0:
                if wrld.monsters_at(x - 1, y): return 1
            if y + 1 < heigth:
                if wrld.monsters_at(x, y + 1): return 1
            if x + 1 < width:
                if wrld.monsters_at(x + 1, y): return 1
            if y - 1 >= 0:
                if wrld.monsters_at(x, y - 1): return 1
            if x - 1 >= 0 and y + 1 < heigth:
                if wrld.monsters_at(x - 1, y + 1): return 1
            if x + 1 < width and y + 1 < heigth:
                if wrld.monsters_at(x + 1, y + 1): return 1
            if x - 1 >= 0 and y - 1 >= 0:
                if wrld.monsters_at(x - 1, y - 1): return 1
            if x + 1 < width and y - 1 >= 0:
                if wrld.monsters_at(x + 1, y - 1): return 1
        return 0

    def nearby_wall(self, wrld):
        me = wrld.me(self)
        if me is not None:
            x = me.x
            y = me.y
            width = wrld.width()
            height = wrld.height()
            if x - 1 >= 0:
                if wrld.wall_at(x - 1, y): return 0
            if y + 1 < height:
                if wrld.wall_at(x, y + 1): return 0
            if x + 1 < width:
                if wrld.wall_at(x + 1, y): return 0
            if y - 1 >= 0:
                if wrld.wall_at(x, y - 1): return 0
            if x - 1 >= 0 and y + 1 < height:
                if wrld.wall_at(x - 1, y + 1): return 0
            if x + 1 < width and y + 1 < height:
                if wrld.wall_at(x + 1, y + 1): return 0
            if x - 1 >= 0 and y - 1 >= 0:
                if wrld.wall_at(x - 1, y - 1): return 0
            if x + 1 < width and y - 1 >= 0:
                if wrld.wall_at(x + 1, y - 1): return 0
        return 1

    # identify if there is a bomb close, and if there is, return the average distance from me to the bombs
    def bomb_threat(self, wrld):
        me = wrld.me(self)
        if me is not None:
            x = me.x
            y = me.y
            width = wrld.width()
            height = wrld.height()

            radius = self.danger_radius

            x_range_high = x + radius
            x_range_low = x - radius

            y_range_high = y + radius
            y_range_low = y - radius

            if x_range_high > width:
                x_range_high = width
            if x_range_low < 0:
                x_range_low = 0
            if y_range_high > height:
                y_range_high = height
            if y_range_low < 0:
                y_range_low = 0

            lob = []  # list of bombs

            for i in range(x_range_low, x_range_high):
                for j in range(y_range_low, y_range_high):
                    if wrld.bomb_at(i, j):
                        lob.append((i, j))

            if len(lob) == 0:
                return 1
            else:
                path = 0
                for bomb in lob:
                    path += len(a_star(wrld, (x, y), (bomb[0], bomb[1])))
                avg = path / len(lob)
                return 0 - (avg / (wrld.height() ** 2 + wrld.width() ** 2) ** 0.5)
        else:
            return 0

    def explosion_threat(self, wrld):
        me = wrld.me(self)
        if me is not None:
            x = me.x
            y = me.y
            width = wrld.width()
            height = wrld.height()

            radius = self.danger_radius

            x_range_high = x + radius
            x_range_low = x - radius

            y_range_high = y + radius
            y_range_low = y - radius

            if x_range_high > width:
                x_range_high = width
            if x_range_low < 0:
                x_range_low = 0
            if y_range_high > height:
                y_range_high = height
            if y_range_low < 0:
                y_range_low = 0

            loe = []  # list of explosions

            for i in range(x_range_low, x_range_high):
                for j in range(y_range_low, y_range_high):
                    if wrld.explosion_at(i, j):
                        loe.append((i, j))

            if len(loe) == 0:
                return 1
            else:
                path = 0
                for explosion in loe:
                    star = a_star(wrld, (x, y), (explosion[0], explosion[1]))
                    if star is not None:
                        path += len(star)
                    else:
                        return 0
                avg = path / len(loe)
                return 0 - (avg / (wrld.height() ** 2 + wrld.width() ** 2) ** 0.5)
        else:
            return 0

    def is_there_monster(self, wrld):
        me = wrld.me(self)
        if me is not None:
            x = me.x
            y = me.y
            width = wrld.width()
            height = wrld.height()
            for i in range(0, height):
                for j in range(0, width):
                    if wrld.monsters_at(i, j):
                        return 0
        return 1

    # This functions selects an action from the list of q_values
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
                elif member == Actions.BOMB:
                    me.place_bomb()
                elif member == Actions.STAY:
                    me.move(0, 0)
                elif member == Actions.FOLLOW_A_STAR:
                    if self.path is None:
                        me.move(0, 0)
                    elif self.path_search <= len(self.path):
                        self.path_search = len(self.path) - 1
                        next_pos = self.path[self.path_search]
                        self.path_search += 1
                        me.move(next_pos[0] - me.x, next_pos[1] - me.y)
                elif member == Actions.RUN_AWAY_BOMB:
                    coord = self.run_away_bomb(wrld)
                    me.move(coord[0] - me.x, coord[1] - me.y)
                list_of_worlds.append((wrld.next(), name))
        else:
            list_of_worlds.append((wrld.next(), "DEAD"))

        return list_of_worlds

    def coord_to_action(self, coord):
        x = self.x
        y = self.y

        if coord == (x, y):
            action = 'STAY'
        elif coord == (x + 1, y):
            action = "RIGHT"
        elif coord == (x + 1, y + 1):
            action = "RIGHT_DOWN"
        elif coord == (x, y + 1):
            action = "DOWN"
        elif coord == (x - 1, y + 1):
            action = "DOWN_LEFT"
        elif coord == (x - 1, y):
            action = "LEFT"
        elif coord == (x - 1, y - 1):
            action = "LEFT_UP"
        elif coord == (x, y - 1):
            action = "UP"
        elif coord == (x + 1, y - 1):
            action = "UP_RIGHT"
        return action

    def run_away_bomb(self, wrld):
        me = wrld.me(self)
        x = me.x
        y = me.y

        width = wrld.width()
        height = wrld.height()

        radius = self.danger_radius

        x_range_high = x + radius
        x_range_low = x - radius

        y_range_high = y + radius
        y_range_low = y - radius

        if x_range_high > width:
            x_range_high = width
        if x_range_low < 0:
            x_range_low = 0
        if y_range_high > height:
            y_range_high = height
        if y_range_low < 0:
            y_range_low = 0

        lob = []  # list of bombs

        for i in range(x_range_low, x_range_high):
            for j in range(y_range_low, y_range_high):
                if wrld.bomb_at(i, j):
                    lob.append((i, j))

        lor = [(-1, (x, y))]  # list of retreats

        for b in lob:
            lor.append(self.run_away(b, wrld))

        retreat = max(lor)

        return retreat[1]

    def run_away_explosion(self, wrld):
        me = wrld.me(self)
        x = me.x
        y = me.y
        width = wrld.width()
        height = wrld.height()

        radius = self.danger_radius

        x_range_high = x + radius
        x_range_low = x - radius

        y_range_high = y + radius
        y_range_low = y - radius

        if x_range_high > width:
            x_range_high = width
        if x_range_low < 0:
            x_range_low = 0
        if y_range_high > height:
            y_range_high = height
        if y_range_low < 0:
            y_range_low = 0

        loe = []  # list of explosions

        for i in range(x_range_low, x_range_high):
            for j in range(y_range_low, y_range_high):
                if wrld.explosion_at(i, j):
                    loe.append((i, j))

        lor = [(-1, (x, y))]  # list of retreats

        for e in loe:
            lor.append(self.avoid_explosion(e, wrld))

        retreat = max(lor)

        return retreat[1]

    def run_away(self, danger_coord, wrld):
        me = wrld.me(self)
        x = me.x
        y = me.y

        flee = [(-1, (0, 0))]  # if all else fails, stand still and accept death

        possibilities = get_adjacent((x, y), wrld)

        for p in possibilities:
            if not wrld.wall_at(p[0], p[1]) and not wrld.monsters_at(p[0], p[1]):
                distance = ((danger_coord[0] - p[0]) ** 2 + (danger_coord[1] - p[1]) ** 2) ** 0.5
                flee.append((distance, p))

        flee = sorted(flee)

        return max(flee)

    def avoid_explosion(self, danger_coord, wrld):
        me = wrld.me(self)
        x = me.x
        y = me.y

        flee = [(-1, (0, 0))]  # if all else fails, stand still and accept death

        possibilities = get_adjacent((x, y), wrld)

        for p in possibilities:
            if not wrld.wall_at(p[0], p[1]) and not wrld.monsters_at(p[0], p[1]):
                if (danger_coord[0] - p[0]) != 0 and (danger_coord[1] - p[1]) != 0:  # not on the same line as bomb
                    distance = ((danger_coord[0] - p[0]) ** 2 + (danger_coord[1] - p[1]) ** 2) ** 0.5
                    flee.append((distance, p))
                else:
                    flee.append((-1, p))

        flee = sorted(flee)

        return max(flee)


# Get a list of all adjacent cells (i.e. not out of bounds)
def get_adjacent(coord, wrld):
    x = coord[0]
    y = coord[1]

    loc = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            if wrld.width() > (x + i) >= 0 and wrld.height() > (y + j) >= 0:
                loc.append((x + i, y + j))

    return loc

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)


def a_star(wrld, start, goal):
    queue = PriorityQueue()
    queue.put(start, 0)
    came_from = {}
    costs = {}
    came_from[start] = None
    costs[start] = 0

    while not queue.empty():
        current = queue.get()
        children = []
        if current == goal:
            break
        (x, y) = current
        if x - 1 >= 0 and not wrld.wall_at(x - 1, y) and not wrld.monsters_at(x - 1, y):
            children.append((x - 1, y))
        if y + 1 < wrld.height() and not wrld.wall_at(x, y + 1) and not wrld.monsters_at(x, y + 1):
            children.append((x, y + 1))
        if x + 1 < wrld.width() and not wrld.wall_at(x + 1, y) and not wrld.monsters_at(x + 1, y):
            children.append((x + 1, y))
        if y - 1 >= 0 and not wrld.wall_at(x, y - 1) and not wrld.monsters_at(x, y - 1):
            children.append((x, y - 1))
        if x - 1 >= 0 and y + 1 < wrld.height() and not wrld.wall_at(x - 1, y + 1) and not wrld.monsters_at(x - 1, y + 1):
            children.append((x - 1, y + 1))
        if x + 1 < wrld.width() and y + 1 < wrld.height() and not wrld.wall_at(x + 1, y + 1) and not wrld.monsters_at(x + 1, y + 1):
            children.append((x + 1, y + 1))
        if x - 1 >= 0 and y - 1 >= 0 and not wrld.wall_at(x - 1, y - 1) and not wrld.monsters_at(x - 1, y - 1):
            children.append((x - 1, y - 1))
        if x + 1 < wrld.width() and y - 1 >= 0 and not wrld.wall_at(x + 1, y - 1) and not wrld.monsters_at(x + 1, y - 1):
            children.append((x + 1, y - 1))
        for child in children:
            cost = costs[current] + 1
            if child not in costs or cost < costs[child]:
                costs[child] = cost
                priority = cost + heuristic(goal, child)
                queue.put(child, priority)
                came_from[child] = current
    if goal not in came_from:
        return came_from
    path = generate_path(came_from, goal)
    return path


def generate_path(came_from, goal):
    current = goal
    path = [goal]
    while current is not None:
        path.append(came_from[current])
        current = came_from[current]
    path.reverse()
    path.remove(None)
    path.pop(0)
    return path


