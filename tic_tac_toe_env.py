
import copy

from mcts_core import MctsEnv


class TicTacToeEnv(MctsEnv):

    def __init__(self, r_seed=0):
        super(TicTacToeEnv, self).__init__(discount=1., action_space=range(0, 9))
        # This is a multiplier in UCB algorithm. 1.0 means no prior.
        self.default_policy_prior = {k:1. for k in range(9)}
        self.r_seed = r_seed

    def reset(self):
        return [0]*9

    def get_predicted_value(self):
        # This is the 'simulation' step in the pure MCTS.
        # TODO: play randomly and get the reward.
        return 0.

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
        for i, state in enumerate(new_states):
            if state == 0:
                new_states[i] = 4
                break

        is_final, reward = self.check(new_states)
        if is_final:
            predicted_value = reward
        else:
            predicted_value = self.get_predicted_value()

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
