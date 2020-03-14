
import unittest

from muzero_eval_policy import MuZeroEvalPolicy
from network_initializer import TicTacToeInitializer
from network import Network
from network import NetworkOutput
from network import Action
from muzero_mcts_model import MuZeroMctsModel
from tic_tac_toe_env import TicTacToeEnv
import tensorflow as tf
import numpy as np
from tic_tac_toe_config import TicTacToeConfig


class MuZeroEvalPolicyTest(unittest.TestCase):
    def setUp(self):
        self.lr = 3e-2
        self.weight_decay = 1e-4
        (self.prediction_network, self.dynamics_network,
         self.representation_network, self.dynamics_encoder,
         self.representation_encoder) = TicTacToeInitializer().initialize()
        self.hidden_state = tf.expand_dims(tf.constant([0] * 9), 0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def scalar_loss(self, y_true, y_pred):
        return tf.square(y_true - y_pred)

    def test_train(self):
        _, predicted_value = self.prediction_network(self.hidden_state)
        with tf.GradientTape() as tape:
            value = tf.constant(1.)
            loss = self.scalar_loss(value, predicted_value)
        gradients = tape.gradient(loss, self.prediction_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.prediction_network.trainable_variables))


if __name__ == '__main__':
    unittest.main()
