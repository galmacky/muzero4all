
import copy
import numpy as np

from mcts_env import MctsEnv


# TODO: split this into TicTacToeEnv and MctsEnvDynamicsModel
class TicTacToeEnv(MctsEnv):

    def __init__(self, r_seed=0):
        super(TicTacToeEnv, self).__init__(discount=1., action_space=range(0, 9))
        # This is a multiplier in UCB algorithm. 1.0 means no prior.
        self.default_policy_prior = {k: 1. for k in range(9)}
        self.r_seed = r_seed

    def reset(self):
        return [0] * 9

    def legal_actions(self, states):
        return [i for i, state in enumerate(states) if state == 0]

    def get_predicted_value(self, states):
        # This is the 'simulation' step in the pure MCTS.
        # TODO: need to check the final states in test for extra clarity.
        states = copy.deepcopy(states)
        while True:
            legal_actions = self.legal_actions(states)
            np.random.seed(self.r_seed)
            self.r_seed += 1
            action = np.random.choice(a=legal_actions)
            states[action] = 1
            is_final, reward = self.check(states)
            if is_final:
                return reward
            opponent_action = self.opponent_play(states)
            states[opponent_action] = 4
            is_final, reward = self.check(states)
            if is_final:
                return reward

    def opponent_play(self, states):
        for i, state in enumerate(states):
            if state == 0:
                return i
        assert False, 'There is no empty space for opponent to play at.'

    def step(self, states, action):
        assert states is not None
        new_states = copy.deepcopy(states)
        # TODO: check if we want to consider legal actions only.
        if new_states[action] != 0:
            return new_states, False, -1., self.default_policy_prior, -1.

        new_states[action] = 1

        is_final, reward = self.check(new_states)
        if is_final:
            return new_states, is_final, reward, self.default_policy_prior, reward

        # TODO(P3): make this smarter (and still reproducible).
        # Opponent places X at the first available space.
        opponent_action = self.opponent_play(new_states)
        new_states[opponent_action] = 4

        is_final, reward = self.check(new_states)
        if is_final:
            predicted_value = reward
        else:
            predicted_value = self.get_predicted_value(new_states)

        return new_states, is_final, reward, self.default_policy_prior, predicted_value

    def check(self, states):
        sums = [sum((states[0], states[1], states[2])), sum((states[3], states[4], states[5])),
                sum((states[6], states[7], states[8])),
                sum((states[0], states[3], states[6])), sum((states[1], states[4], states[7])),
                sum((states[2], states[5], states[8])),
                sum((states[0], states[4], states[8])), sum((states[2], states[4], states[7]))]
        if 3 in sums:
            return True, 1. # win
        if 12 in sums:
            return True, -1. # loss
        if 0 in sums:
            return False, 0. # not finished
        return True, 0. # draw
