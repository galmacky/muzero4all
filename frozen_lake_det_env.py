
import copy
import gym

from mcts_env import MctsEnv

# coding: utf-8
"""Defines some frozen lake maps."""
from gym.envs.toy_text import frozen_lake, discrete
from gym.envs.registration import register

register(
    id='Deterministic-4x4-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False})

# Deterministic Frozen Lake
class FrozenLakeEnv(MctsEnv):

    def __init__(self):
        super(FrozenLakeEnv, self).__init__(action_space=range(0, 9))
        # This is a multiplier in UCB algorithm. 1.0 means no prior.
        self.default_policy_prior = {k: 1. for k in range(9)}

        # 4x4, deterministic due to is_slipper=False
        self.env = gym.make("Deterministic-4x4-FrozenLake-v0")
        seed = 1234
        self.env.seed(seed)

        # List of actions
        self._states = []


    def reset(self):
        self.env.reset()

    def get_states(self):
        return self._states

    # Set a state by going through all of the actions in states, starting
    # from the beginning.
    def set_states(self, states):
        """states is a list of actions."""
        self._states = states
        self.reset()
        for action in states:
            self.step(action)

    # Take a step and append to the list of actions stored in self._states
    def step(self, action):
        # ob is unused
        ob, rew, done , _ = self.env.step(action)

        self._states.append(action)
        new_states = self._states

        return new_states, done, rew