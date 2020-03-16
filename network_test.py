
import unittest
from network_initializer import TicTacToeInitializer
from network import Network, NetworkOutput, Action
from muzero_mcts_model import MuZeroMctsModel
from tic_tac_toe_env import TicTacToeEnv
import tensorflow as tf
import numpy as np
from tic_tac_toe_config import TicTacToeConfig

class NetworkTest(unittest.TestCase):
    def setUp(self):
        # Make tests reproducible.
        np.random.seed(0)
        tf.random.set_seed(0)
        self.env = TicTacToeEnv()
        self.network_initializer = TicTacToeInitializer()
        self.network = Network(self.network_initializer)
        self.model = MuZeroMctsModel(self.env, self.network)

    def test_initial_inference(self):
        game_state = np.array(self.env.get_states()).reshape(-1, TicTacToeConfig.action_size)
        output = self.network.initial_inference(game_state)
        self.assertTrue(output.reward == 0)
        # self.assertTrue(output.value.shape == (1, 2*TicTacToeConfig.support_size + 1))
        self.assertTrue(output.value.shape == (1, TicTacToeConfig.value_size))
        self.assertTrue(output.policy_logits.shape == (1, TicTacToeConfig.action_size))
        self.assertTrue(output.hidden_state.shape == (1, TicTacToeConfig.hidden_size))

    def test_recurrent_inference(self):
        game_state = np.array(self.env.get_states()).reshape(-1, TicTacToeConfig.action_size)
        action = Action(0)
        output = self.network.recurrent_inference(game_state, action)
        #self.assertEqual(0, output.reward)
        # self.assertTrue(output.value.shape == (1, 2*TicTacToeConfig.support_size + 1))
        self.assertTrue(output.value.shape == (1, TicTacToeConfig.value_size))
        self.assertTrue(output.policy_logits.shape == (1, TicTacToeConfig.action_size))
        self.assertTrue(output.hidden_state.shape == (1, TicTacToeConfig.hidden_size))


if __name__ == '__main__':
    unittest.main()
