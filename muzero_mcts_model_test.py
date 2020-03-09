
import unittest


class MuZeroMctsModelTest(unittest.TestCase):

    def setUp(self):
        self.network_initializer = NetworkInitializer()
        self.network = Network()
        self.model = MuZeroMctsModel(env, self.network)

    def test_basic(self):
        pass
