
import unittest

from mcts_env_dynamics_model import MctsEnvDynamicsModel
from tic_tac_toe_env import TicTacToeEnv


class MctsEnvDynamicsModelTest(unittest.TestCase):

    def setUp(self):
        self.env = TicTacToeEnv()
        self.dynamics_model = MctsEnvDynamicsModel(self.env)

    def test_get_predicted_value_and_final_info(self):
        # Check some conditions first.
        states = [0] * 9
        states[4] = 1
        states[0] = 4
        self.assertEqual((False, 0.0), self.env.check(states))
        self.assertEqual(1, self.env.opponent_play(states))

        predicted_value, final_info = self.dynamics_model.get_predicted_value_and_final_info(states)
        self.assertEqual(-1., predicted_value)
        # Game ended for an illegal move.
        self.assertEqual([4, 4, 0, 0, 1, 1, 0, 0, 0], final_info[0])  # states
        self.assertEqual(5, final_info[1])  # action

    # TODO: add a test for discount != 1.0
