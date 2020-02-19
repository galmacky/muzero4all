
import unittest

from mcts_env_dynamics_model import MctsEnvDynamicsModel
from mcts_policy import MctsPolicy
from tic_tac_toe_env import TicTacToeEnv


class MctsPolicyTest(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()
        self.dynamics_model = MctsEnvDynamicsModel(self.env)
        self.policy = MctsPolicy(self.env, self.dynamics_model)

    def test_basic(self):
        # TODO: add more detailed tests.
        action = self.policy.action()
        self.assertEqual(3, action)
