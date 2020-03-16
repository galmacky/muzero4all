
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

    def get_random_states(self):
        # Returns tic-tac-toe states
        return np.square(np.random.randint(low=0, high=3, size=(10, 9)))

    def bak_test_update_weights(self):
        self.batch = [
            # prediction:
            1.,
            (self.get_random_states(), [0, 1], [0]),
            (self.get_random_states(), [0, 1, 2], [0]),
            (self.get_random_states(), [0, 1, 2, 3], [0]),
            (self.get_random_states(), [0, 1, 2, 3, 4], [0]),
        ]
        loss = 0.

        class MyModel(tf.keras.Model):

            def __init__(self):
                super(MyModel, self).__init__()
                self.prediction_network = PredictionNetwork()
                self.dynamics_network = DynamicsNetwork()
                self.representation_network = RepresentationNetwork()
                self.dynamics_encoder = DynamicsEncoder()
                self.representation_encoder = RepresentationEncoder()

            def call(self, inputs):
                first_output = self.prediction_network(inputs)
                return first_output

        self.model = MyModel()

        for image, actions, targets in self.batch:
            # Reshape the states to be -1 x n dimension: -1 being the outer batch dimension.
            image = np.array(image).reshape(-1, len(image))
            # Initial step, from the real observation.
            value, reward, policy_logits, hidden_state = self.network.initial_inference(
                image)
            predictions = [(1.0, value, reward, policy_logits)]

            # Recurrent steps, from action and previous hidden state.
            for action in actions:
                value, reward, policy_logits, hidden_state = self.network.recurrent_inference(
                    hidden_state, Action(action))
                predictions.append((1.0 / len(actions), value, reward, policy_logits))

            for prediction, target in zip(predictions, targets):
                gradient_scale, value, reward, policy_logits = prediction

                target_value, target_reward, target_policy = target
                # TODO: fix reward / target_reward to be float32.

        losses = (layers.losses.MSE, layers.losses.MSE, layers.losses.cross)
        # self.optimizer.minimize(lambda: loss, var_list=self.network.get_weights())
        self.network.compile()
        self.network.fit()
        print('loss', loss)


if __name__ == '__main__':
    unittest.main()
