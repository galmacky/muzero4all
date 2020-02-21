
import copy

from mcts_env import MctsEnv


class TicTacToeEnv(MctsEnv):

    def __init__(self):
        super(TicTacToeEnv, self).__init__(action_space=range(0, 9))
        # This is a multiplier in UCB algorithm. 1.0 means no prior.
        self.default_policy_prior = {k: 1. for k in range(9)}
        self._states = [0] * 9

    def reset(self):
        self._states = [0] * 9
        return self._states

    def legal_actions(self, states):
        return [i for i, state in enumerate(states) if state == 0]

    def opponent_play(self, states):
        for i, state in enumerate(states):
            if state == 0:
                return i
        assert False, 'There is no empty space for opponent to play at.'

    def set_states(self, states):
        self._states = states

    def get_states(self):
        return self._states

    def step(self, action):
        # TODO: check if we want to consider legal actions only.
        if self._states[action] != 0:
            return self._states, True, -1.

        self._states[action] = 1

        is_final, reward = self.check(self._states)
        if is_final:
            return self._states, is_final, reward

        # TODO(P3): make this smarter (and still reproducible).
        # Opponent places X at the first available space.
        opponent_action = self.opponent_play(self._states)
        self._states[opponent_action] = 4

        is_final, reward = self.check(self._states)
        return self._states, is_final, reward

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
