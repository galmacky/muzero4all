
import unittest

from tensorflow.keras import datasets, layers, models
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
        self.network = models.Sequential()
        self.network.add(layers.Dense(9, activation='relu'))
        self.network.add(layers.Dense(1, activation='relu'))
        self.hidden_state = tf.expand_dims(tf.constant([0] * 9), 0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def scalar_loss(self, y_true, y_pred):
        return tf.square(y_true - y_pred)

    def test_train(self):
        predicted_value = self.network(self.hidden_state)
        with tf.GradientTape() as tape:
            value = tf.constant(1.)
            loss = self.scalar_loss(value, predicted_value)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))


if __name__ == '__main__':
    unittest.main()
