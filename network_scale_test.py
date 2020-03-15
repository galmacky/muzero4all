
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
        # Make test results reproducible.
        np.random.seed(0)
        tf.random.set_seed(0)

    def test_train_single_sequential(self):
        self.model = models.Sequential([
            # layers.Flatten(input_shape=(9,)),
            layers.Dense(9, activation='relu'),
            # layers.Dropout(0.2),
            # Alternatively, you could pass activation=None.
            layers.Dense(1, activation='relu')
        ])

        self.state = tf.expand_dims(tf.constant([0] * 9), 0)
        data = np.random.randint(low=0, high=2, size=(10, 9))
        self.state = tf.constant(data)

        self.loss_fn = tf.keras.losses.MSE
        self.model.compile(optimizer='adam', loss=self.loss_fn, metrics=['mae'])
        actual_value = tf.constant([[.9]] * 10)
        self.model.fit(self.state, actual_value, epochs=300, verbose=0)
        loss, _ = self.model.evaluate(self.state, actual_value, verbose=0)
        self.assertLess(loss, 1e-02)

    def test_train_multiple_sequential(self):
        class MyModel(tf.keras.Model):

            def __init__(self):
                super(MyModel, self).__init__()
                self.first_network = models.Sequential([
                    layers.Dense(9, activation='relu'),
                    layers.Dense(1, activation='relu')
                ])
                self.second_network = models.Sequential([
                    layers.Dense(9, activation='relu'),
                    layers.Dense(1, activation='relu')
                ])

            def call(self, inputs):
                first_output = self.first_network(inputs)
                second_output = self.second_network(inputs)
                return first_output, second_output

        self.model = MyModel()

        self.state = tf.expand_dims(tf.constant([0] * 9), 0)
        data = np.random.randint(low=0, high=2, size=(10, 9))
        self.state = tf.constant(data)

        self.loss_fn = tf.keras.losses.MSE
        self.model.compile(optimizer='adam',
                           loss=[self.loss_fn, self.loss_fn],
                           loss_weights=[1., 1.],
                           metrics=['mae'])
        actual_value = tf.constant([[.9]] * 10)
        train_y = [actual_value, actual_value]
        self.model.fit(self.state,
                       train_y,
                       epochs=300, verbose=0)
        # Note: evaluate does not support dict.
        test_data = [actual_value, actual_value]
        loss = self.model.evaluate(self.state, test_data, verbose=2)
        self.assertEqual(5, len(loss))  # loss, output_1_loss, output_2_loss, output_1_mae, output_2_mae
        # Decent but slower than single network.
        self.assertLess(loss[0], 1e-01)


if __name__ == '__main__':
    unittest.main()
