# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
from Q_WEIGHTS import Q_WEIGHTS

class TestCharacter(CharacterEntity):

    def do(self, wrld):
        # Your code here

        # test = Q_WEIGHTS
        #
        # print(test)
        #
        # test[1] = test[1] + 1
        #
        # f = open("../Q_WEIGHTS.py", "w")
        #
        # f.write("Q_WEIGHTS = " + str(test))
        #
        # f.close()

        print(self.distance_to_exit(wrld))

        pass

    def distance_to_exit(self, wrld):
        x = self.x
        y = self.y
        x_d = wrld.exitcell[0]
        y_d = wrld.exitcell[1]
        return ((x-x_d)**2 + (y-y_d)**2)**.5
