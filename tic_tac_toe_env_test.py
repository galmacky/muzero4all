
import unittest

from tic_tac_toe_env import TicTacToeEnv


class TicTacToeEnvTest(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()

    def test_check(self):
        self.assertEqual((False, 0.), self.env.check([0] * 8 + [1]))
        self.assertEqual((True, -1.), self.env.check([4, 4, 4, 0, 1, 1, 0, 0, 0]))
        self.assertEqual((True, 1.), self.env.check([4, 4, 0, 1, 1, 1, 0, 0, 0]))

    def test_legal_actions(self):
        states = [0] * 9
        states[3] = 1
        states[7] = 4
        states[8] = 1
        self.assertEqual([0, 1, 2, 4, 5, 6], self.env.legal_actions(states))

    def test_opponent_play(self):
        # Chooses the first available space.
        self.assertEqual(0, self.env.opponent_play([0] * 8 + [1]))
        self.assertEqual(8, self.env.opponent_play([1] * 8 + [0]))

    def test_step(self):
        self.env.set_states([4, 4, 0, 0, 1, 1, 0, 0, 0])
        states, is_final, reward = self.env.step(3)
        self.assertEqual([4, 4, 0, 1, 1, 1, 0, 0, 0], states)
        self.assertTrue(is_final)
        self.assertEqual(1., reward)


if __name__ == '__main__':
    unittest.main()
