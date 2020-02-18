
import unittest

from tic_tac_toe_env import TicTacToeEnv


class TicTacToeEnvTest(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()

    def test_check(self):
        self.assertEquals((False, 0.0), self.env.check([0] * 8 + [1]))

    def test_legal_actions(self):
        states = [0] * 9
        states[3] = 1
        states[7] = 4
        states[8] = 1
        self.assertEquals([0, 1, 2, 4, 5, 6], self.env.legal_actions(states))

    def test_opponent_play(self):
        # Chooses the first available space.
        self.assertEquals(0, self.env.opponent_play([0] * 8 + [1]))
        self.assertEquals(8, self.env.opponent_play([1] * 8 + [0]))

    def test_get_predicted_value(self):
        # Check some conditions first.
        states = [0] * 9
        states[4] = 1
        states[0] = 4
        self.assertEquals((False, 0.0), self.env.check(states))
        self.assertEquals(1, self.env.opponent_play(states))

        self.assertEquals(-1., self.env.get_predicted_value(states))

if __name__ == '__main__':
    unittest.main()
