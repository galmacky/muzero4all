import abc

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from network_architectures import ResnetIdentityBlock 

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
			super(PredictionNetwork, self).__init__()
			#Define model here
			hidden_size = 64
			action_size = 9 #3x3 board
			support_size = 10
			self.policy_network = models.Sequential()
			self.policy_network.add(layers.Dense(hidden_size, activation='relu'))
			self.policy_network.add(layers.Dense(action_size, activation='relu'))
			
			self.value_network = models.Sequential()
			self.value_network.add(layers.Dense(hidden_size, activation='relu'))
			self.value_network.add(layers.Dense(2*support_size+1, activation='relu'))

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
			super(PredictionNetwork, self).__init__()
			representation_size = 64
			hidden_size = 64
			action_size = 9 #3x3 board

			self.dynamic_network = models.Sequential()
			self.dynamic_network.add(layers.Dense(representation_size, activation='relu'))
			self.dynamic_network.add(layers.Dense(hidden_size, activation='relu'))
			
			self.reward_network = models.Sequential()
			self.reward_network.add(layers.Dense(representation_size, activation='relu'))
			self.reward_network.add(layers.Dense(1, activation='relu'))

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
			super(PredictionNetwork, self).__init__()
			representation_size = 64
			hidden_size = 64
			action_size = 9 #3x3 board
			self.representation_network = models.Sequential()
			self.representation_network.add(layers.Dense(representation_size, activation='relu'))
			self.representation_network.add(layers.Dense(hidden_size, activation='relu'))

		def call(self, inputs):
			hidden_state = representation_network(inputs)
			return hidden_state

	def initialize(self):
		prediction_network = PredictionNetwork()
		dynamics_network = DynamicNetwork()
		representation_network = RepresentationNetwork()
		return (prediction_network, dynamics_network, representation_network)

'''
	Builds the dynamics, representation, and prediction
	models for playing FrozenLake games.
'''
class FrozenLake(NetworkInitializer):

	def initialize(self):
		prediction_network = models.Sequential()
		prediction_network.add(layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))
		prediction_network.add(layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))
		prediction_network.add(layers.Dense(output_size)) #TODO(FJUR): put correct output size

		dynamics_network = models.Sequential()
		dynamics_network.add(layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))
		dynamics_network.add(layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))
		dynamics_network.add(layers.AveragePooling2D(pool_size=(3,3), strides=(3, 3)))
		dynamics_network.add(layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))

		representation_network = models.Sequential()
		representation_network.add(layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))
		representation_network.add(layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))
		representation_network.add(layers.AveragePooling2D(pool_size=(3,3), strides=(3, 3)))
		representation_network.add(layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))

		return (prediction_network, dynamics_network, representation_network)
	

'''
	Builds the dynamics, representation, and prediction
	models for playing Atari games.
'''
class AtariInitializer(NetworkInitializer):

	def initialize(self):
		# : one or two convolutional layers
		# that preserve the resolution but reduce the number of planes, followed by a fully connected layer to the size of the
		# output.
		prediction_network = models.Sequential()
		prediction_network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
		prediction_network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
		prediction_network.add(layers.Dense(output_size))

		dynamics_network = models.Sequential()
		# 1 convolution with stride 2 and 128 output planes, output resolution 48x48.
		dynamics_network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
		# 2 residual blocks with 128 planes
		dynamics_network.add(ResnetIdentityBlock((3,3), [128, 128, 128]))
		dynamics_network.add(ResnetIdentityBlock((3,3), [128, 128, 128]))
		# 1 convolution with stride 2 and 256 output planes, output resolution 24x24
		dynamics_network.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
		# 3 residual blocks with 256 planes.
		dynamics_network.add(ResnetIdentityBlock((3,3), [256, 256, 256]))
		dynamics_network.add(ResnetIdentityBlock((3,3), [256, 256, 256]))
		dynamics_network.add(ResnetIdentityBlock((3,3), [256, 256, 256]))
		# Average pooling with stride 2, output resolution 12x12.
		dynamics_network.add(layers.AveragePooling2D(pool_size=(3,3), strides=(3, 3)))
		dynamics_network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

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
		# 1 convolution with stride 2 and 128 output planes, output resolution 48x48.
		representation_network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
		# 2 residual blocks with 128 planes
		representation_network.add(ResnetIdentityBlock((3,3), [128, 128, 128]))
		representation_network.add(ResnetIdentityBlock((3,3), [128, 128, 128]))
		# 1 convolution with stride 2 and 256 output planes, output resolution 24x24
		representation_network.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
		representation_network.add(ResnetIdentityBlock((3,3), [256, 256, 256]))
		representation_network.add(ResnetIdentityBlock((3,3), [256, 256, 256]))
		representation_network.add(ResnetIdentityBlock((3,3), [256, 256, 256]))
		# Average pooling with stride 2, output resolution 12x12.
		representation_network.add(layers.AveragePooling2D(pool_size=(3,3), strides=(3, 3)))
		representation_network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
		'''
		1 convolution with stride 2 and 128 output planes, output resolution 48x48
		2 residual blocks with 128 planes
		1 convolution with stride 2 and 256 output planes, output resolution 24x24.
		3 residual blocks with 256 planes.
		Average pooling with stride 2, output resolution 12x12.
		3 residual blocks with 256 planes.
		Average pooling with stride 2, output resolution 6x6.
		The kernel size is 3x3 for all operations.
		'''

		return (prediction_network, dynamics_network, representation_network)
