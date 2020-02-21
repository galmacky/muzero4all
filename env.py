
import abc


class Env(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, action_space):
        self.action_space = action_space

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def step(self, action):
        """Returns (states, is_final, reward).
        Note: states will eventually get called bet into set_states()
        """
        pass
