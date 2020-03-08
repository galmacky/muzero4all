
import gym
import re

from mcts_env import MctsEnv

# coding: utf-8
from gym.envs.registration import register

# register(
#     id='Deterministic-4x4-FrozenLake-v0',
#     entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
#     kwargs={'map_name': '4x4',
#             'is_slippery': False})

# Deterministic Pacman
class PacmanDetEnv(MctsEnv):

    def __init__(self, screen_images=False):
        if screen_images:
            # This is using screen images (may be better for muzero).
            self.env = gym.make("MsPacmanDeterministic-v0")
        else:
            # Deterministic using RAM as input (more efficient) for pure MCTS.
            self.env = gym.make("MsPacman-ramDeterministic-v0")
        seed = 1234
        self.env.seed(seed)

        # List of past historical actions.
        # For example, can be called onto set_states to reset the env and
        # cause all past actions to be taken (assuming deterministic envs)
        # resulting in a desired end state after replaying all actions.
        self._states = []

        # Number of actions (Aliased by a map from int -> string
        # https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py#L219).
        nA = self.env.action_space.n

        super(PacmanDetEnv, self).__init__(action_space=range(nA))

    def reset(self):
        self._states = []
        self.env.reset()
        return self._states

    def get_states(self):
        return self._states

    def get_real_states(self):
        """Returns env's real states not the action history."""
        return self.remove_ansi(self.env.render(mode='ansi').getvalue())

    def remove_ansi(self, ansi_str):
        ansi_escape_8bit = re.compile(
            r'(?:\x1B[@-Z\\-_]|[\x80-\x9A\x9C-\x9F]|(?:\x1B\[|\x9B)[0-?]*[ -/]*[@-~])'
        )
        return ansi_escape_8bit.sub('', ansi_str)

    def set_states(self, states):
        """Set the given states.

        Set a state by going through all of the actions in states, starting
        from the beginning.

        Args:
            states: A list of actions to get to the states.
        """
        self.reset()
        for action in states:
            self.step(action)

    def step(self, action):
        # TODO: remove this once we fix action to be non-tensor.
        action = int(action)
        _, rew, done , _ = self.env.step(action)

        self._states.append(action)
        new_states = self._states

        return new_states, done, rew