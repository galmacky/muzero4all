
import unittest
from network_initializer import TicTacToeInitializer
from network import Network, NetworkOutput, Action
from muzero_mcts_model import MuZeroMctsModel
from tic_tac_toe_env import TicTacToeEnv
import tensorflow as tf
import numpy as np
from tic_tac_toe_config import TicTacToeConfig

class TicTacToeNetworkInitializerTest(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()
        self.network_initializer = TicTacToeInitializer()
        self.prediction_network, self.dynamics_network, self.representation_network, self.dynamics_encoder = self.network_initializer.initialize()

    def test_prediction_network(self):
        input_image = np.array(self.env.get_states()).reshape(-1, TicTacToeConfig.action_size)
        policy_logits, value = self.prediction_network(input_image)
        # self.assertTrue(value.shape == (1, 2*TicTacToeConfig.support_size + 1))
        self.assertTrue(value.shape == (1, TicTacToeConfig.value_size))
        self.assertTrue(policy_logits.shape == (1, TicTacToeConfig.action_size))
       
    def test_representation_network(self):
        input_image = np.array(self.env.get_states()).reshape(-1, TicTacToeConfig.action_size)
        hidden_state = self.representation_network(input_image)
        
        self.assertTrue(hidden_state.shape == (1, TicTacToeConfig.hidden_size))

        # self.assertEqual(output.value, np.zeros([1, 2*support_size + 1]))
        # self.assertTrue(output.reward == 0)
        # self.assertTrue(output.reward == 0)

    def test_dynamics_network(self):
        hidden_state = np.zeros((TicTacToeConfig.batch_size, TicTacToeConfig.hidden_size))
        action = Action(0)
        encoded_state = self.dynamics_encoder.encode(hidden_state, action)
        hidden_state, reward = self.dynamics_network(encoded_state)
        self.assertTrue(reward == 0)
        self.assertTrue(hidden_state.shape == (1, TicTacToeConfig.hidden_size))

    def test_encoded_dynamics_state(self):
        hidden_state = np.zeros((TicTacToeConfig.batch_size, TicTacToeConfig.hidden_size))
        action = Action(0)
        hidden_state = np.zeros((TicTacToeConfig.batch_size, TicTacToeConfig.hidden_size))
        encoded_state = self.dynamics_encoder.encode(hidden_state, action)
        #TODO(FJUR): This should encode 2 planes, a 1 hot plane with the selected action,
        # and a binary plane whether the move was valid or not (all 0's or 1's)
        pass