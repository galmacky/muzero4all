import typing

from network_initializer import NetworkInitializer
import numpy as np

class Action(object):
    """ Class that represent an action of a game."""

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index

'''
    Interface for the output of the Network
'''
class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: typing.Dict[Action, float]
    hidden_state: typing.List[float]

'''
    Generic network class, pass in the initializer for game model (Tic Tac Toe, or Atari)
    to build the model.
'''
class Network(object):

    def __init__(self, initializer: NetworkInitializer):
        self.prediction_network, self.dynamics_network, self.representation_network, self.dynamics_encoder, self.represetnation_encoder = initializer.initialize()
        self.training_steps = 0

    def get_all_trainable_weights(self):
        return (self.prediction_network.trainable_weights + 
        self.dynamics_network.trainable_weights + 
        self.representation_network.trainable_weights)

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        hidden_state = self.representation_network(image)
        policy_logits, value = self.prediction_network(hidden_state)
        return NetworkOutput(value, 0, policy_logits, hidden_state)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        #Need to encode action information with hidden state before passing
        #to the dynamics function.
        encoded_state = self.dynamics_encoder.encode(hidden_state, action)
        hidden_state, reward = self.dynamics_network(encoded_state)
        policy_logits, value = self.prediction_network(hidden_state)
        #Enable this when value/reward are discrete support sets.
        # value = _decode_support_set(value)
        return NetworkOutput(value, reward, policy_logits, hidden_state)

    def get_weights(self):
        # Returns the weights of this network.
        # return np.concatenate(self.representation_network.get_weights(), self.dynamics_network.get_weights(), self.prediction_network.get_weights())
        # return np.concatenate([self.representation_network.get_weights(), self.dynamics_network.get_weights(), self.prediction_network.get_weights()]).ravel()
        return self.get_all_trainable_weights()
        # weights = []
        # for weight in self.representation_network.get_weights():
        #     weights.append(weights)
        # for weight in self.dynamics_network.get_weights():
        #     weights.append(weights)
        # for weight in self.prediction_network.get_weights():
        #     weights.append(weights)

        # return weights
    
    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.training_steps

    #Private
    def _decode_support_set(self, logits):
        """
        Converts discrete set of size support*2 + 1 to a scalar.
        This is used for value and reward.
        """
        value = tf.nn.softmax(logits)
        value = np.dot(value, range(self.value_support_size))
        value = self._inverse_invertible_transform(value)
        return value

    def _encode_support_set(self, logits):
        """
        Converts a scalar to a discrete set of size support*2 + 1.
        This is used for value and reward.
        """
        pass

    # From the MuZero paper.
    def _invertible_transform(x, eps=0.001):
        return tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1.) - 1.) + eps * x

    #Private
    # From the MuZero paper.
    def _inverse_invertible_transform(x, eps=0.001):
        return tf.math.sign(x) * (
            tf.math.square(
                (tf.sqrt(4 * eps *
                        (tf.math.abs(x) + 1. + eps) + 1.) - 1.) / (2. * eps)) - 1.)