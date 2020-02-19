
from mcts_core import MctsCore


class MctsPolicy(object):

    def __init__(self, env, dynamics_model, num_simulations=100):
        self.core = MctsCore(env)
        self.env = env
        self.dynamics_model = dynamics_model
        self.num_simulations = num_simulations

    def reset(self):
        self.dynamics_model.reset()

    def action(self):
        self.core.initialize()
        for _ in range(self.num_simulations):
            self.core.rollout()
        policy_logits = self.core.get_policy_distribution()
