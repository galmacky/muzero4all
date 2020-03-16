
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


class PredictionNetwork(tf.keras.Model):
    '''
    Creates a network that returns the policy logits and the value
    returns : policy_logits, value
    '''

    def __init__(self):
        super(PredictionNetwork, self).__init__()
        # Define model here
        self.policy_network = models.Sequential()
        self.policy_network.add(layers.Dense(TicTacToeConfig.hidden_size, activation='relu'))
        self.policy_network.add(layers.Dense(TicTacToeConfig.action_size, activation='relu'))

        self.value_network = models.Sequential()
        self.value_network.add(layers.Dense(TicTacToeConfig.hidden_size, activation='relu'))
        # self.value_network.add(layers.Dense(TicTacToeConfig.support_size *2 + 1, activation='relu'))
        self.value_network.add(layers.Dense(TicTacToeConfig.value_size, activation='relu'))

    def call(self, inputs):
        policy_logits = self.policy_network(inputs)
        value = self.value_network(inputs)
        return (policy_logits, value)


class DynamicsNetwork(tf.keras.Model):
    '''
    Given the current hidden state and action, transition to the next hidden state given action.
    inputs: hidden_state: current hidden state of muzero algorithm, action: action taken from current hidden state)
    returns: hidden_state: next hidden state, reward: reward from current hidden state.

    Actions are encoded spatially in planes of the same resolution as the hidden state. In Atari, this resolution is 6x6 (see description of downsampling in Network Architecture section), in board games this is the same as the board size (19x19 for Go, 8x8 for chess, 9x9 for shogi). 1
    '''

    def __init__(self):
        super(DynamicsNetwork, self).__init__()
        self.dynamic_network = models.Sequential()
        self.dynamic_network.add(layers.Dense(TicTacToeConfig.representation_size, activation='relu'))
        self.dynamic_network.add(layers.Dense(TicTacToeConfig.hidden_size, activation='relu'))

        self.reward_network = models.Sequential()
        self.reward_network.add(layers.Dense(TicTacToeConfig.representation_size, activation='relu'))
        self.reward_network.add(layers.Dense(TicTacToeConfig.reward_size, activation='relu'))

    '''
    Input is hidden state concat 2 one hot encodings planes of 9x9. 1 hot for action in tic tac toe, 1 for if valid.
    '''

    def call(self, inputs):
        next_hidden_state = self.dynamic_network(inputs)
        reward = self.reward_network(inputs)
        return (next_hidden_state, reward)


class RepresentationNetwork(tf.keras.Model):
    '''
    Converts the initial state of the gameboard to the muzero hidden state representation.
    inputs: initial state
    returns: hidden state
    '''

    def __init__(self):
        super(RepresentationNetwork, self).__init__()
        self.representation_network = models.Sequential()
        self.representation_network.add(layers.Dense(TicTacToeConfig.representation_size, activation='relu'))
        self.representation_network.add(layers.Dense(TicTacToeConfig.hidden_size, activation='relu'))

    def call(self, inputs):
        hidden_state = self.representation_network(inputs)
        return hidden_state


class DynamicsEncoder(object):
    def encode(self, hidden_state, action):
        encoded_actions = tf.one_hot(action.index, TicTacToeConfig.action_size)
        encoded_actions = tf.expand_dims(encoded_actions, 0)
        encoded_hidden_state = tf.concat([hidden_state, encoded_actions], axis=0)
        encoded_hidden_state = tf.expand_dims(encoded_hidden_state, 0)
        # Tic Tac Toe uses dense layer so flatten.
        encoded_hidden_state = tf.expand_dims(tf.reshape(encoded_hidden_state, [-1]), 0)
        return encoded_hidden_state


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

            def call(self, inputs):
                first_output = self.prediction_network(inputs)
                return first_output

            def initial_inference(self, image) -> NetworkOutput:
                # representation + prediction function
                hidden_state = self.representation_network(image)
                policy_logits, value = self.prediction_network(hidden_state)
                return NetworkOutput(value, 0, policy_logits, hidden_state)

            def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
                # dynamics + prediction function
                # Need to encode action information with hidden state before passing
                # to the dynamics function.
                encoded_state = self.dynamics_encoder.encode(hidden_state, action)
                hidden_state, reward = self.dynamics_network(encoded_state)
                policy_logits, value = self.prediction_network(hidden_state)
                # Enable this when value/reward are discrete support sets.
                # value = _decode_support_set(value)
                return NetworkOutput(value, reward, policy_logits, hidden_state)

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
