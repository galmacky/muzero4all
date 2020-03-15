
import numpy as np

from mcts_env import MctsEnv

class SimpleEnv(MctsEnv):

    def __init__(self):
        super(SimpleEnv, self).__init__(action_space=range(0, 2))
        self._states = [0] * 2
    
    def render(self):
        print("")
        print("------")
        print("{}|{}|".format(self._states[0],self._states[1]))
        print("------")

    def reset(self):
        self._states = [0] * 2
        return self._states

    def set_states(self, states):
        self._states = states

    def get_states(self):
        return self._states

    def step(self, action):
        if action == 0:
            return self._states, True, 1
        else:
            return self._states, True, -1

    def get_current_game_input(self):
        return np.array(self._states)
        #return ''.join([str(pos_rep) for pos_rep in self._states])

    def check(self, states):
        sums = [sum((states[0], states[1], states[2])), sum((states[3], states[4], states[5])),
                sum((states[6], states[7], states[8])),
                sum((states[0], states[3], states[6])), sum((states[1], states[4], states[7])),
                sum((states[2], states[5], states[8])),
                sum((states[0], states[4], states[8])), sum((states[2], states[4], states[6]))]
        if 3 in sums:
            return True, 1. # win
        if 12 in sums:
            return True, -1. # loss
        if 0 in states:
            return False, 0. # not finished
        return True, 0. # draw
