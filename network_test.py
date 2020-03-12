
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
        self.env = TicTacToeEnv()
        self.network_initializer = TicTacToeInitializer()
        self.network = Network(self.network_initializer)
        self.model = MuZeroMctsModel(self.env, self.network)

    def test_initial_inference(self):
        game_state = np.array(self.env.get_states()).reshape(-1, TicTacToeConfig.action_size)
        output = self.network.initial_inference(game_state)
        self.assertTrue(output.reward == 0)
        self.assertTrue(output.value.shape == (1, 2*TicTacToeConfig.support_size + 1))
        self.assertTrue(output.policy_logits.shape == (1, TicTacToeConfig.action_size))
        self.assertTrue(output.hidden_state.shape == (1, TicTacToeConfig.hidden_size))

    def test_recurrent_inference(self):
        game_state = np.array(self.env.get_states()).reshape(-1, TicTacToeConfig.action_size)
        action = Action(0)
        output = self.network.recurrent_inference(game_state, action)
        self.assertTrue(output.reward == 0)
        self.assertTrue(output.value.shape == (1, 2*TicTacToeConfig.support_size + 1))
        self.assertTrue(output.policy_logits.shape == (1, TicTacToeConfig.action_size))
        self.assertTrue(output.hidden_state.shape == (1, TicTacToeConfig.hidden_size))
       
