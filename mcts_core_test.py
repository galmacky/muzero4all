
import unittest

from mcts_core import MctsEnv
from mcts_core import MctsCore
from mcts_core import Node

class TicTacToeEnv(MctsEnv):

  def __init__(self):
    super(TicTacToeEnv, self).__init__(discount=1., action_space=range(0, 9))

  def step(self, states, action):
    if states[action] != 0:
      return 
  

class MctsCoreTest(unittest.TestCase):

  def setUp(self):
    self.env = TicTacToeEnv()
    self.core = MctsCore(num_simulations=1, env=self.env)

  def test(self):
    self.core.initialize([0]*9)
    self.core.rollout()
    # TODO: check nodes after rollout
    self.core.rollout()
    # TODO: check nodes after rollout


if __name__ == '__main__':
    unittest.main()
