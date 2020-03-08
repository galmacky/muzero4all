import typing

from network_initializer import NetworkInitializer

#This is just a placeholder
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
        (prediction_network, dynamics_network, representation_network) = initializer.initialize()
        self.representation = representation_network
        self.dynamics = dynamics_network
        self.prediction = prediction_network

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        hidden_state = self.representation(image)
        #Pretty sure  self.prediction doesn't return two things
        policy_logits, value = self.prediction(hidden_state)
        return NetworkOutput(value, 0, policy_logits, hidden_state)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        hidden_state, reward  = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(hidden_state)
        return NetworkOutput(value, reward, policy_logits, hidden_state)

    def get_weights(self):
        # Returns the weights of this network.
        return []

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return 0
