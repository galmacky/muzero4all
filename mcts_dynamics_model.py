
import abc

class MctsDynamicsModel(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def step(self, states, action):
        pass


class MctsEnvDynamicsModel(MctsDynamicsModel):

    def __init__(self, action_space, env, r_seed=0):
        self.action_space = action_space
        # This is a multiplier in UCB algorithm. 1.0 means no prior.
        self.default_policy_prior = {k: 1. for k in action_space}
        self.env = env
        self.r_seed = r_seed

    def step(self, states, action):
        pass
