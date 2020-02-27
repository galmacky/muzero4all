
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
        # 4x4, deterministic due to is_slipper=False.
        self.env = gym.make("Deterministic-4x4-FrozenLake-v0")
        seed = 1234
        self.env.seed(seed)

        # List of past historical actions.
        # For example, can be called onto set_states to reset the env and
        # cause all past actions to be taken (assuming deterministic envs as we are 
        # using deterministic frozen lake) resulting in a desired end state after 
        # replaying all actions.
        self._states = []

        # Aliased by constants in frozen lake (https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py)
        # Number of actions (Frozen lake is discrete).
        nA = self.env.action_space.n

        # TODO(changwan): Does is this how you want aciton_space to be defined? 
        # just a list of ints?????????????????????

        super(FrozenLakeEnv, self).__init__(action_space=range(nA))


    def reset(self):
        self._states = []
        self.env.reset()

    def get_states(self):
        return self._states

    # Set a state by going through all of the actions in states, starting
    # from the beginning.
    def set_states(self, states):
        """states is a list of actions."""
        self._states = states
        self.reset()
        # self.step() will append to self._states unneccessarily
        for action in states:
            self.env.step(int(action))

    # Take a step and append to the list of actions stored in self._states.
    def step(self, action):
        # ob is unused
        action = int(action)
        ob, rew, done , _ = self.env.step(action)

        self._states.append(action)
        new_states = self._states

        return new_states, done, rew