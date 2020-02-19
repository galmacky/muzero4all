
import copy
import numpy as np

from mcts_dynamics_model import MctsDynamicsModel


class MctsEnvDynamicsModel(MctsDynamicsModel):
    """Dynamics model for pure MCTS."""

    def __init__(self, env, discount=1., max_depth=100, r_seed=0):
        # This is a multiplier in UCB algorithm. 1.0 means no prior.
        self.default_policy_prior = {k: 1. for k in env.action_space}
        self.env = env
        self.discount = discount
        self.max_depth = max_depth
        self.r_seed = r_seed

    def get_initial_states(self):
        return self.env.reset()

    def step(self, states, action):
        new_states, is_final, reward = self.env.step(states, action)
        if is_final:
            predicted_value = reward
        else:
            predicted_value = self.get_predicted_value(new_states)
        return new_states, is_final, reward, self.default_policy_prior, predicted_value

    def get_predicted_value(self, starting_states):
        depth = 0
        states = copy.deepcopy(starting_states)
        # This is the 'simulation' step in the pure MCTS.
        returns = 0.
        acc_discount = 1.
        while True:
            np.random.seed(self.r_seed)
            self.r_seed += 1
            action = np.random.choice(a=self.env.action_space)

            states, is_final, reward = self.env.step(states, action)
            returns += acc_discount * reward
            acc_discount *= self.discount
            depth += 1

            if is_final or depth >= self.max_depth:
                # TODO: need to check the final states in test for extra clarity.
                return returns
