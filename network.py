import typing

from network_initializer import NetworkInitializer

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
        prediction_network, dynamics_network, representation_network, dynamics_encoder = initializer.initialize()
        self.representation_network = representation_network
        self.dynamics_network = dynamics_network
        self.prediction_network = prediction_network
        self.dynamics_encoder = dynamics_encoder
        self.training_steps = 0

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
        return NetworkOutput(value, reward, policy_logits, hidden_state)

    def get_weights(self):
        # Returns the weights of this network.
        return [self.representation_network.get_weights(), self.dynamics_network.get_weights(), self.prediction_network.get_weights()]

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.training_steps
