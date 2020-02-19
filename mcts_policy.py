
from mcts_core import MctsCore


class MctsPolicy(object):

    def __init__(self, env, dynamics_model):
        self.core = MctsCore(env)
        self.env = env
        self.dynamics_model = dynamics_model

    def action(self):
        # TODO
        init_states = None
        self.core.initialize()
