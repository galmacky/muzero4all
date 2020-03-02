
import unittest

from basic_mcts_model import BasicMctsModel
from tic_tac_toe_env import TicTacToeEnv


class BasicMctsModelTest(unittest.TestCase):

    def setUp(self):
        self.env = TicTacToeEnv()
        self.dynamics_model = BasicMctsModel(self.env)

    def test_get_predicted_value_and_final_info_discounted(self):
        self.dynamics_model = BasicMctsModel(self.env, discount=0.9)
        # Check some conditions first.
        states = [0] * 9
        states[4] = 1
        states[0] = 4
        self.assertEqual((False, 0.0), self.env.check(states))
        self.assertEqual(1, self.env.opponent_play(states))

        predicted_value, final_info = self.dynamics_model.get_predicted_value_and_final_info(states)
        self.assertEqual(-.9, predicted_value)
        # Game ended for an illegal move.
        self.assertEqual([4, 4, 0, 0, 1, 1, 0, 0, 0], final_info[0])  # states
        self.assertEqual(5, final_info[1])  # action

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

    def test_step(self):
        self.dynamics_model.step([0] * 9, 0)
        # The above is a simulation step, so it should not affect the real environment.
        self.assertEqual([0] * 9, self.env.get_states())

    # TODO: add a test for discount != 1.0


if __name__ == '__main__':
    unittest.main()
