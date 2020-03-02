
import abc


class MctsModel(object):
    """This is a plug-in architecture for MCTS core. TODO: the name is somewhat misleading. Rename this."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_initial_states(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def step(self, states, action):
        """
            Args:
                states: For pure MCTS, this should be the states of the model. For MuZero, this is the hidden states.
                action: An action.

            Returns: (new_states, is_final, immediate_reward, policy_prior_dict, predicted_value_for_new_states)

        """
        pass
