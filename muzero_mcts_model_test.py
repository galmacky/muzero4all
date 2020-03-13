
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
        hidden_state = self.model.get_initial_states()
        tf.assert_equal(tf.constant(tf.zeros((1, 64))), hidden_state)
        model_step = self.model.step(hidden_state, Action(1))
        # Not trained, so returning an empty states.
        tf.assert_equal(tf.constant(tf.zeros((1, 64))), model_step[0])  # states
        self.assertFalse(model_step[1])  # is_final
        self.assertEqual(0.0, model_step[2])
        # Note: Policy logits are all empty. This could be problematic since it affects initial
        # exploration.
        tf.assert_equal(tf.constant(tf.zeros((1, 9))), model_step[3])  # policy
        # TODO: this should be a single value?
        tf.assert_equal(tf.constant(tf.zeros((1, 21))), model_step[4])  # value


if __name__ == '__main__':
    unittest.main()
