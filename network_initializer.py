import abc

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# class Model(object):
#     """This is an environment for Mcts Environment.
#     """
#     __metaclass__ = abc.ABCMeta

#     def __init__(self, action_space):
#         super(MctsEnv, self).__init__(action_space)

#     @abc.abstractmethod
#     def set_states(self, states):
#         pass

'''
A base abstract class passed to network.Network which will
initialize the required models.
'''
class NetworkInitializer(object):
     __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def initialize(self) -> List:
        pass

# class TicTacToeInitializer(object):

#     def initialize

'''
Builds the dynamics, representation, and prediction
models for playing Atari games.
'''
class AtariInitializer(NetworkInitializer):



    def initialize(self) -> List:
        dynamics_network
        # model = models.Sequential()
        # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        # model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # layers = [tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid) for _ in range(n)]
        # perceptron = tf.keras.Sequential(layers)

        # Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
        representation_network = models.Sequential()
        representation_network = models.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')
        '''
        • 1 convolution with stride 2 and 128 output planes, output resolution 48x48.
        • 2 residual blocks with 128 planes
        • 1 convolution with stride 2 and 256 output planes, output resolution 24x24.
        • 3 residual blocks with 256 planes.
        • Average pooling with stride 2, output resolution 12x12.
        • 3 residual blocks with 256 planes.
        • Average pooling with stride 2, output resolution 6x6.
        The kernel size is 3x3 for all operations.
        '''