
import abc


class MctsEnv(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, discount, action_space):
        self.discount = discount
        self.action_space = action_space

    @abc.abstractmethod
    def step(self, states, action):
        pass

    def legal_actions(self):
        return self.action_space


