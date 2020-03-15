
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


class NetworkScaleTest(unittest.TestCase):
    def setUp(self):
        pass

    def scalar_loss(self, y_true, y_pred):
        return tf.square(y_true - y_pred)

    def test_train_1(self):
        self.lr = 3e-2
        self.weight_decay = 1e-4
        self.network = models.Sequential()
        self.network.add(layers.Dense(9, activation='relu'))
        self.network.add(layers.Dense(1, activation='relu'))
        self.state = tf.expand_dims(tf.constant([0] * 9), 0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.network.compile(optimizer='adam', loss=self.loss_fn, metrics=['accuracy'])
        predicted_value = self.network(self.state)
        actual_value = tf.constant([[.9]])
        print (predicted_value, actual_value)
        self.network.fit(self.state, actual_value, epochs=20)


if __name__ == '__main__':
    unittest.main()
