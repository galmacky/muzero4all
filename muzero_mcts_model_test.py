
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
        # Set seeds to make the results reproducible.
        np.random.seed(0)
        tf.random.set_seed(0)
        self.env = TicTacToeEnv()
        self.network_initializer = TicTacToeInitializer()
        self.network = Network(self.network_initializer)
        self.model = MuZeroMctsModel(self.env, self.network)

    def test_basic(self):
        hidden_state = self.model.get_initial_states()
        tf.assert_equal(tf.constant(tf.zeros((1, 9))), hidden_state)
        model_step = self.model.step(hidden_state, 1)
        # Not trained, so returning an empty states.
        self.assertEqual([1, 9], model_step[0].shape)  # states
        self.assertFalse(model_step[1])  # is_final is always false in MuZero MCTS
        self.assertEqual(0.0, model_step[2])
        self.assertEqual([9], model_step[3].shape)  # policy
        # TODO: remove the outer dimension.
        self.assertEqual([1, 1], model_step[4].shape)  # value


if __name__ == '__main__':
    unittest.main()
