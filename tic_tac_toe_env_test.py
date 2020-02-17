
import unittest

from tic_tac_toe_env import TicTacToeEnv


class TicTacToeEnvTest(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()

    def testBasic(self):
        pass


if __name__ == '__main__':
    unittest.main()
