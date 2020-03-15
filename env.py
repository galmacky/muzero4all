
import abc

from typing import List

class Env(object):
    """An environment for MuZero that also serves as base class for pure MCTS environment."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, action_space):
        self.action_space = action_space

    @abc.abstractmethod
    def reset(self) -> List:
        pass

    # TODO: rename this to observation
    @abc.abstractmethod
    def get_current_game_input(self):
        pass

    @abc.abstractmethod
    def step(self, action):
        """Returns (states, is_final, reward).
        Note: states will eventually get called bet into set_states()
        """
        pass
