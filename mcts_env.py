
import abc


class MctsEnv(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, action_space):
        self.action_space = action_space

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def set_states(self):
        pass

    @abc.abstractmethod
    def step(self, action):
        pass
