
import unittest
from network_initializer import TicTacToeInitializer
from network import Network
from network import NetworkOutput
from network import Action
from muzero_mcts_model import MuZeroMctsModel
from tic_tac_toe_env import TicTacToeEnv
import tensorflow as tf
import numpy as np
from tic_tac_toe_config import TicTacToeConfig


class MuZeroMctsModelTest(unittest.TestCase):

    def setUp(self):
        self.env = TicTacToeEnv()
        self.network_initializer = TicTacToeInitializer()
        self.network = Network(self.network_initializer)
        self.model = MuZeroMctsModel(self.env, self.network)

    def test_basic(self):
        pass
        #self.assertEqual(tf.constant([0] * 9), self.model.get_initial_states())
        #self.assertEqual([], self.model.step([0] * 9, 1))


if __name__ == '__main__':
    unittest.main()
