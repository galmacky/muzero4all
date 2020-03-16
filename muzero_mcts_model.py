
import numpy as np
import tensorflow as tf

from env import Env
from mcts_model import MctsModel
from network import Action
from network import Network


class MuZeroMctsModel(MctsModel):
    """MCTS model for MuZero algorithm."""

    def __init__(self, env: Env, network: Network):
        self.env = env
        self.network = network

    def set_network(self, network):
        self.network = network

    def get_initial_states(self):
        # TOOD: FIX THIS NEEDS TO USE OTHER METHOD!!!!!!!!!!!
        # Reshape the states to be -1 x n dimension: -1 being the outer batch dimension.
        inputs = self.env.get_current_game_input()
        game_state = inputs.reshape(-1, len(inputs))
        # Note: we only use the initial hidden states. Other information will be used in a subsequent 'step' method.
        output = self.network.initial_inference(game_state)
        return output.hidden_state

    def reset(self):
        # This is called when we call policy.reset()
        self.network.reset()

    def step(self, parent_states, action):
        output = self.network.recurrent_inference(parent_states, Action(action))
        # Note: we do not have is_final value. This can cause a serious error.
        policy_logits = tf.reshape(output.policy_logits, [len(self.env.action_space)])
        # print('Step action: ', action)
        # print('Step reward: ', output.reward.numpy())
        return output.hidden_state, output.reward == -1, output.reward, policy_logits, output.value
