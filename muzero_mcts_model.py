
from mcts_model import MctsModel


class MuZeroMctsModel(MctsModel):

    def __init__(self, env, network):
        self.env = env
        self.network = network

    def set_network(self, network):
        self.network = network

    def get_initial_states(self):
        # TODO
        return [0] * 9
        return self.network.encode(self.env.get_states())

    def reset(self):
        # This is called when we call policy.reset()
        self.network.reset()
        pass

    def step(self, states, action):
        return [0] * 9, False, 0.0, {i: 1.0 for i in range(0, 9)}, 0.1
        # TODO: return (new_states, is_final, immediate_reward, policy_prior_dict, predicted_value_for_new_states)
        return self.network.virtual_step(states)
