
from mcts_dynamics_model import MctsDynamicsModel


class MuZeroDynamicsModel(MctsDynamicsModel):

    def __init__(self, env, network):
        self.env = env
        self.network = network

    def set_network(self, network):
        self.network = network

    def get_initial_states(self):
        return self.network.encode(self.env.get_states())

    def reset(self):
        # This is called when we call policy.reset()
        self.network.reset()
        pass

    def step(self, states, action):
        # TODO: return (new_states, is_final, immediate_reward, policy_prior_dict, predicted_value_for_new_states)
        return self.network.virtual_step(states)
