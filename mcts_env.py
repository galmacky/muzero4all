# Env for just pure MCTS (requires perfect simulator, thus
# needs set_states).

import abc

from env import Env

class MctsEnv(Env):
    """This is an environment for pure Mcts Environment."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, action_space):
        super(MctsEnv, self).__init__(action_space)

    @abc.abstractmethod
    def get_states(self):
        pass

    @abc.abstractmethod
    def set_states(self, states):
        pass
