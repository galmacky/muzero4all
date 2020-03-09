
import gym
import re

from mcts_env import MctsEnv

# coding: utf-8
from gym.envs.registration import register

# Deterministic Pacman
class PacmanDetEnv(MctsEnv):

    def __init__(self, screen_images=False, 
        use_clone_state=True):
        self.use_clone_state = use_clone_state
        if screen_images:
            # This is using screen images (may be better for muzero).
            self.env = gym.make("MsPacmanDeterministic-v0")
        else:
            # Deterministic using RAM as input (more efficient) for pure MCTS.
            self.env = gym.make("MsPacman-ramDeterministic-v0")
        self.env.reset()

        seed = 1234
        self.env.seed(seed)

        # List of past historical actions.
        # For example, can be called onto set_states to reset the env and
        # cause all past actions to be taken (assuming deterministic envs)
        # resulting in a desired end state after replaying all actions.
        if self.use_clone_state:
            self._states = self.env.unwrapped.clone_full_state()
        else:
            self._states = []

        # Number of actions (Aliased by a map from int -> string
        # https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py#L219).
        nA = self.env.action_space.n

        super(PacmanDetEnv, self).__init__(action_space=range(nA))

    def reset(self):
        self.env.reset()
        if self.use_clone_state:
            self._states = self.env.unwrapped.clone_full_state()
        else:
            self._states = []
        return self._states

    def get_states(self):
        return self._states

    # START REMOVE?????????????????
    # def get_real_states(self):
    #     """Returns env's real states not the action history."""
    #     return self.remove_ansi(self.env.render(mode='ansi').getvalue())

    # def remove_ansi(self, ansi_str):
    #     ansi_escape_8bit = re.compile(
    #         r'(?:\x1B[@-Z\\-_]|[\x80-\x9A\x9C-\x9F]|(?:\x1B\[|\x9B)[0-?]*[ -/]*[@-~])'
    #     )
    #     return ansi_escape_8bit.sub('', ansi_str)
    # END REMOVE??????????????????

    def set_states(self, states):
        """Set the given states.

        Set a state by going through all of the actions in states, starting
        from the beginning.

        Args:
            states: A list of actions to get to the states.
        """
        if self.use_clone_state:
            self.env.unwrapped.restore_full_state(states)
            self._states = states
        else:
            self.reset()
            for action in states:
                self.step(action)

    def step(self, action):
        # TODO: remove this once we fix action to be non-tensor.
        action = int(action)
        _, rew, done , _ = self.env.step(action)

        if self.use_clone_state:
            new_states = self.env.unwrapped.clone_full_state()
            self._states = new_states
        else:
            self._states.append(action)
            new_states = self._states

        return new_states, done, rew