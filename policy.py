
import abc


class Policy(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def action(self):
        """
            Returns:
                An action
        """
        pass

