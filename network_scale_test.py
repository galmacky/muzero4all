
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
        self.seed = 0
        self.lr = 3e-2
        self.weight_decay = 1e-4
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        self.model = models.Sequential([
            layers.Flatten(input_shape=(9,)),
            layers.Dense(9, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation=None)
        ])

        self.state = tf.expand_dims(tf.constant([0] * 9), 0)
        data = np.random.randint(low=0, high=2, size=(10, 9))
        print (data)
        self.state = tf.constant(data)

        self.loss_fn = tf.keras.losses.MSE
        self.model.compile(optimizer='adam', loss=self.loss_fn, metrics=['mae'])
        predicted_value = self.model(self.state)
        actual_value = tf.constant([[.9]] * 10)
        print (predicted_value, actual_value)
        self.model.fit(self.state, actual_value, epochs=100, verbose=0)
        loss, _ = self.model.evaluate(self.state, actual_value, verbose=0)
        self.assertAlmostEqual(0.13333006, loss)


if __name__ == '__main__':
    unittest.main()
