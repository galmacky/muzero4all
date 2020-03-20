import abc

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from network_architectures import ResnetIdentityBlock 
from tic_tac_toe_config import TicTacToeConfig
from pac_man_config import PacManConfig
from atari_config import AtariConfig
import numpy as np

'''
A base abstract class passed to network.Network which will
initialize the required models.
'''
class NetworkInitializer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def initialize(self):
        pass

'''
    Builds the dynamics, representation, and prediction
    models for playing Tic Tac Toe games.
'''
class TicTacToeInitializer(NetworkInitializer):

    class PredictionNetwork(tf.keras.Model):
        '''
        Creates a network that returns the policy logits and the value
        returns : policy_logits, value 
        '''
        def __init__(self):
            super(TicTacToeInitializer().PredictionNetwork, self).__init__()
            #Define model here
            self.policy_network = models.Sequential()
            self.policy_network.add(layers.Dense(TicTacToeConfig.hidden_size, activation='relu'))
            self.policy_network.add(layers.Dense(TicTacToeConfig.action_size, activation='relu'))
            
            self.value_network = models.Sequential()
            self.value_network.add(layers.Dense(TicTacToeConfig.hidden_size, activation='relu'))
            self.value_network.add(layers.Dense(TicTacToeConfig.value_size, activation='tanh'))

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
            super(TicTacToeInitializer().DynamicsNetwork, self).__init__()
            self.dynamic_network = models.Sequential()
            self.dynamic_network.add(layers.Dense(TicTacToeConfig.representation_size, activation='relu'))
            self.dynamic_network.add(layers.Dense(TicTacToeConfig.hidden_size, activation=None))
            
            self.reward_network = models.Sequential()
            self.reward_network.add(layers.Dense(TicTacToeConfig.representation_size, activation='relu'))
            self.reward_network.add(layers.Dense(TicTacToeConfig.reward_size, activation='tanh'))
        
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
            super(TicTacToeInitializer().RepresentationNetwork, self).__init__()
            self.representation_network = models.Sequential()
            self.representation_network.add(layers.Dense(TicTacToeConfig.representation_size, activation='relu'))
            self.representation_network.add(layers.Dense(TicTacToeConfig.hidden_size, activation='tanh'))
        
        def call(self, inputs):
            hidden_state = self.representation_network(inputs)
            return hidden_state

    class DynamicsEncoder(object):
        def encode(self, hidden_state, action):
            encoded_actions = tf.one_hot(action.index, TicTacToeConfig.action_size)
            encoded_actions = tf.expand_dims(encoded_actions, 0)
            encoded_hidden_state= tf.concat([hidden_state, encoded_actions], axis=0) 
            encoded_hidden_state = np.expand_dims(encoded_hidden_state, 0)
            #Tic Tac Toe uses dense layer so flatten.
            encoded_hidden_state = tf.expand_dims(tf.reshape(encoded_hidden_state, [-1]), 0)
            return encoded_hidden_state

    class RepresentationEncoder(object):
        def encode(self, hidden_state, action):
            action_plane = tf.one_hot(actions_batch, TicTacToeConfig.action_size)
            # Recurrent step from conditioned representation: recurrent + prediction networks
            representation_stack = tf.concat((representation_batch, action_plane), axis=1)
            return representation_stack

    def initialize(self):
        prediction_network = TicTacToeInitializer().PredictionNetwork()
        dynamics_network = TicTacToeInitializer().DynamicsNetwork()
        representation_network = TicTacToeInitializer().RepresentationNetwork()
        dynamics_encoder = TicTacToeInitializer().DynamicsEncoder()
        representation_encoder = TicTacToeInitializer().RepresentationEncoder()
        return (prediction_network, dynamics_network, representation_network, dynamics_encoder, representation_encoder)


'''
    Builds the dynamics, representation, and prediction
    models for playing Ms Pacman games.
'''
class PacManInitializer(NetworkInitializer):

    class PredictionNetwork(tf.keras.Model):
        '''
        Creates a network that returns the policy logits and the value
        returns : policy_logits, value 
        '''
        def __init__(self):
            super(PacManInitializer.PredictionNetwork, self).__init__()
            #Define model here
            self.policy_network = models.Sequential()
            self.policy_network.add(layers.Dense(PacManConfig.hidden_size, activation='relu'))
            self.policy_network.add(layers.Dense(PacManConfig.action_size, activation='relu'))
            
            self.value_network = models.Sequential()
            self.value_network.add(layers.Dense(PacManConfig.hidden_size, activation='relu'))
            # self.value_network.add(layers.Dense(PacManConfig.support_size *2 + 1, activation='relu'))
            self.value_network.add(layers.Dense(PacManConfig.value_size, activation='relu'))

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
            super(PacManInitializer().DynamicsNetwork, self).__init__()
            self.dynamic_network = models.Sequential()
            self.dynamic_network.add(layers.Dense(PacManConfig.representation_size, activation='relu'))
            self.dynamic_network.add(layers.Dense(PacManConfig.hidden_size, activation='relu'))
            
            self.reward_network = models.Sequential()
            self.reward_network.add(layers.Dense(PacManConfig.representation_size, activation='relu'))
            self.reward_network.add(layers.Dense(PacManConfig.reward_size, activation='relu'))
        
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
            super(PacManInitializer().RepresentationNetwork, self).__init__()
            self.representation_network = models.Sequential()
            self.representation_network.add(layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))
            self.representation_network.add(layers.Flatten())
            self.representation_network.add(layers.Dense(PacManConfig.representation_size, activation='relu'))
            self.representation_network.add(layers.Dense(PacManConfig.hidden_size, activation='relu'))
        
        def call(self, inputs):
            hidden_state = self.representation_network(inputs)
            return hidden_state

    class DynamicsEncoder(object):
        def encode(self, hidden_state, action):
            #[[...], 0, 0, 0, 0,]
            #Doing a onehot of size hidden_size, only actually onehotting from
            #0 - action_size, then from action_size to hidden size is 0 padding.
            #This works because we are using a flatten.
            encoded_actions = tf.one_hot(action.index, PacManConfig.hidden_size)
            encoded_actions = tf.expand_dims(encoded_actions, 0)
            encoded_hidden_state= tf.concat([hidden_state, encoded_actions], axis=0) 
            encoded_hidden_state = np.expand_dims(encoded_hidden_state, 0)
            #Pacman uses dense layer so flatten.
            encoded_hidden_state = tf.expand_dims(tf.reshape(encoded_hidden_state, [-1]), 0)
            return encoded_hidden_state

    class RepresentationEncoder(object):
        def encode(self, hidden_state, action):
            return hidden_state
          
    def initialize(self):
        prediction_network = PacManInitializer().PredictionNetwork()
        dynamics_network = PacManInitializer().DynamicsNetwork()
        representation_network = PacManInitializer().RepresentationNetwork()
        dynamics_encoder = PacManInitializer().DynamicsEncoder()
        representation_encoder = PacManInitializer().RepresentationEncoder()
        return (prediction_network, dynamics_network, representation_network, dynamics_encoder, representation_encoder)

'''
    Builds the dynamics, representation, and prediction
    models for playing Atari games.
'''
class AtariInitializer(NetworkInitializer):
    class PredictionNetwork(tf.keras.Model):
        '''
        Creates a network that returns the policy logits and the value
        returns : policy_logits, value
        '''
        def __init__(self):
            super(AtariInitializer.PredictionNetwork, self).__init__()
            self.policy_network = models.Sequential()
            self.policy_network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
            self.policy_network.add(ResnetIdentityBlock((3,3), [128, 128, 128]))
            self.policy_network.add(layers.Flatten())
            self.policy_network.add(layers.Dense(128, activation='relu'))
            self.policy_network.add(layers.Dense(AtariConfig.action_size))

            self.value_network = models.Sequential()
            self.value_network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
            self.value_network.add(ResnetIdentityBlock((3,3), [128, 128, 128]))
            self.value_network.add(layers.Flatten())
            self.value_network.add(layers.Dense(128, activation='relu'))
            self.value_network.add(layers.Dense(AtariConfig.value_size))

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
            super(AtariInitializer().DynamicsNetwork, self).__init__()
            self.dynamic_network = models.Sequential()
            self.dynamic_network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
            self.dynamic_network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
            self.dynamic_network.add(layers.AveragePooling2D(pool_size=(3,3), strides=(3, 3), padding='same'))
            self.dynamic_network.add(ResnetIdentityBlock((3,3), [128, 128, 128]))
            self.dynamic_network.add(layers.AveragePooling2D(pool_size=(3,3), strides=(3, 3), padding='same'))

            self.reward_network = models.Sequential()
            self.reward_network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
            self.reward_network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
            self.reward_network.add(layers.AveragePooling2D(pool_size=(3,3), strides=(3, 3), padding='same'))
            self.reward_network.add(ResnetIdentityBlock((3,3), [128, 128, 128]))
            self.reward_network.add(layers.AveragePooling2D(pool_size=(3,3), strides=(3, 3), padding='same'))
            self.reward_network.add(layers.Flatten())
            self.reward_network.add(layers.Dense(128, activation='relu'))
            self.reward_network.add(layers.Dense(AtariConfig.reward_size))
        
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
            super(AtariInitializer().RepresentationNetwork, self).__init__()
            self.representation_network = models.Sequential()
            self.representation_network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
            self.representation_network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
            self.representation_network.add(layers.AveragePooling2D(pool_size=(3,3), strides=(3, 3), padding='same'))
            self.representation_network.add(ResnetIdentityBlock((3,3), [128, 128, 128]))
            self.representation_network.add(layers.AveragePooling2D(pool_size=(3,3), strides=(3, 3), padding='same'))
        
        def call(self, inputs):
            hidden_state = self.representation_network(inputs)
            return hidden_state

    class DynamicsEncoder(object):
        def encode(self, hidden_state, action):
            encoded_actions = tf.one_hot(action.index, 22*17)
            encoded_actions = tf.reshape(encoded_actions, [22, 17, 1])
            encoded_actions = tf.expand_dims(encoded_actions, 0)
            encoded_hidden_state= tf.concat([hidden_state, encoded_actions], axis=3)
            return encoded_hidden_state

    class RepresentationEncoder(object):
        def encode(self, hidden_state, action):
            return hidden_state

    def initialize(self):
        prediction_network = AtariInitializer().PredictionNetwork()
        dynamics_network = AtariInitializer().DynamicsNetwork()
        representation_network = AtariInitializer().RepresentationNetwork()
        dynamics_encoder = AtariInitializer().DynamicsEncoder()
        representation_encoder = AtariInitializer().RepresentationEncoder()
        return (prediction_network, dynamics_network, representation_network, dynamics_encoder, representation_encoder)
