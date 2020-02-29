
import abc


class Policy(object):
    """A policy that returns a action for the given environment and state."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def action(self):
        """
            Returns:
                An action
        """
        pass

