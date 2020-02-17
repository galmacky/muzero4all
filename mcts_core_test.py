
import copy
import unittest

from mcts_core import MctsEnv
from mcts_core import MctsCore
from mcts_core import Node


# TODO: move to another file
# TODO: add tests
class TicTacToeEnv(MctsEnv):

  def __init__(self):
    super(TicTacToeEnv, self).__init__(discount=1., action_space=range(0, 9))
    self.default_policy_prior = {k:1. for k in range(9)}

  def reset(self):
    return [0]*9

  def step(self, states, action):
    assert states is not None
    new_states = copy.deepcopy(states)
    if new_states[action] != 0:
      return new_states, -1., {}, -1.

    new_states[action] = 1

    is_final, reward = self.check(new_states)
    if is_final:
      return new_states, is_final, reward, self.default_policy_prior, reward

    # TODO(P3): make this smarter (and still reproducible).
    # Opponent places X at the first available space.
    for i, state in enumerate(new_states):
      if state == 0:
        new_states[i] = 4

    is_final, reward = self.check(new_states)

    return new_states, is_final, reward, self.default_policy_prior, reward

  def check(self, states):
    sums = [sum((states[0], states[1], states[2])), sum((states[3], states[4], states[5])),
            sum((states[6], states[7], states[8])),
            sum((states[0], states[3], states[6])), sum((states[1], states[4], states[7])),
            sum((states[2], states[5], states[8])),
            sum((states[0], states[4], states[8])), sum((states[2], states[4], states[7]))]
    if 3 in sums:
      return True, 1. # win
    if 12 in sums:
      return True, -1. # loss
    if 0 in sums:
      return False, 0. # not finished
    return True, 0. # draw


class MctsCoreTicTacToeTest(unittest.TestCase):

  def setUp(self):
    self.env = TicTacToeEnv()
    self.core = MctsCore(num_simulations=1, env=self.env)

  def test_game(self):
    self.core.initialize()
    # TODO: run tic-tac-toe game 100 times and check winning ratios.

  def test_rollout(self):
    self.core.initialize()
    self.core.rollout()
    self.assertEqual('{v: 1, p: 1.0, v_sum: -1.0, s: [0, 0, 0, 0, 0, 0, 0, 0, 0], r: 0, c: {0: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 1: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 2: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 3: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 4: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 5: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 6: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 7: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 8: {v: 1, p: 1.0, v_sum: -1.0, s: None, r: -1.0, c: {0: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 1: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 2: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 3: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 4: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 5: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 6: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 7: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 8: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}}}}}',
        str(self.core.get_root_for_testing()))
    self.core.rollout()
    # TODO: check nodes after rollout

  def test_inside_initial_rollouts(self):
    self.core.initialize()
    root = self.core.get_root_for_testing()

    node1, search_path, last_action = self.core.select_node(root)

    self.assertNotEqual(root, node1)
    self.assertEqual([root, node1], search_path)
    # TODO: why? analyze and comment (or fix)
    self.assertEqual(8, last_action)

    self.core.expand_node(node1)

    # TODO: test ucb distributions of first few rollouts


if __name__ == '__main__':
    unittest.main()
