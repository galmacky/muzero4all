
import copy
import numpy as np

from mcts_dynamics_model import MctsDynamicsModel


class MctsEnvDynamicsModel(MctsDynamicsModel):
    """Dynamics model for pure MCTS."""

    def __init__(self, env, discount=1., max_depth=100, r_seed=0):
        # This is a multiplier in UCB algorithm. 1.0 means no prior.
        self.default_policy_prior = {k: 1. for k in env.action_space}
        self.env = env
        # TODO: check if we need to discount more
        self.discount = discount
        self.max_depth = max_depth
        self.r_seed_init = r_seed
        self.r_seed = r_seed

    def get_initial_states(self):
        return self.env.reset()

    def reset(self):
        self.r_seed = self.r_seed_init

    def step(self, states, action):
        self.env.set_states(states)
        new_states, is_final, reward = self.env.step(action)
        if is_final:
            predicted_value = reward
        else:
            predicted_value, _ = self.get_predicted_value_and_final_info(new_states)
        return new_states, is_final, reward, self.default_policy_prior, predicted_value

    def get_predicted_value_and_final_info(self, starting_states):
        depth = 0
        states = copy.deepcopy(starting_states)
        # This is the 'simulation' step in the pure MCTS.
        returns = 0.
        acc_discount = 1.
        while True:
            np.random.seed(self.r_seed)
            self.r_seed += 1
            action = np.random.choice(a=self.env.action_space)

            self.env.set_states(states)
            states, is_final, reward = self.env.step(action)
            returns += acc_discount * reward
            acc_discount *= self.discount
            depth += 1

            if is_final or depth >= self.max_depth:
                self.env.set_states(starting_states)
                return returns, (states, action)
