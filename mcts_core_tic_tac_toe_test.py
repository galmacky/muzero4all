
import numpy as np
import unittest

from mcts_core import MctsCore
from basic_mcts_model import BasicMctsModel
from tic_tac_toe_env import TicTacToeEnv


class MctsCoreTicTacToeTest(unittest.TestCase):

    def setUp(self):
        self.env = TicTacToeEnv()
        self.dynamics_model = BasicMctsModel(self.env)
        self.core = MctsCore(env=self.env, dynamics_model=self.dynamics_model)

    def test_rollout(self):
        self.core.initialize()
        # TODO: test more succinctly.
        self.assertEqual('{v: 0, p: 1.0, v_sum: 0, s: [0, 0, 0, 0, 0, 0, 0, 0, 0], r: 0, c: {0: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 1: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 2: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 3: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 4: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 5: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 6: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 7: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 8: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}}}',
                         str(self.core.get_root_for_testing()))
        self.core.rollout()
        # TODO: test more succinctly.
        self.assertEqual('{v: 1, p: 1.0, v_sum: -1.0, s: [0, 0, 0, 0, 0, 0, 0, 0, 0], r: 0, c: {0: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 1: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 2: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 3: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 4: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 5: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 6: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 7: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 8: {v: 1, p: 1.0, v_sum: -1.0, s: [4, 0, 0, 0, 0, 0, 0, 0, 1], r: 0.0, c: {0: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 1: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 2: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 3: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 4: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 5: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 6: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 7: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 8: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}}}}}',
                         str(self.core.get_root_for_testing()))
        # rollout should not affect the actual test.
        self.assertEqual([0] * 9, self.env.get_states())
        self.core.rollout()
        # TODO: check nodes after rollout

    def get_ucb_distribution(self, node):
        # A list of (ucb, action, child).
        l1 = list(self.core.get_ucb_distribution(node))
        l2 = [(action, ucb) for ucb, action, child in l1]
        l3 = sorted(l2)
        # Returns the list of ucb in the order of action.
        return [ucb for _, ucb in l3]

    def test_inside_initial_rollouts(self):
        self.core.initialize()
        root = self.core.get_root_for_testing()

        self.assertEqual([0.] * 9, self.get_ucb_distribution(root))

        node1, search_path, last_action = self.core.select_node()

        self.assertNotEqual(root, node1)
        self.assertEqual([root, node1], search_path)
        # We can choose any action since the ucb distribution is uniform over actions.
        self.assertEqual(8, last_action)

        self.core.expand_node(node1)

        self.assertTrue(node1.expanded())
        parent = root
        self.assertIsNotNone(parent.states)

        value = self.core.evaluate_node(node1, parent.states, last_action)

        self.assertEqual(0., node1.reward)
        # Opponent (4) placed X at the first empty space.
        self.assertEqual([4] + [0] * 7 + [1], node1.states)
        # It is likely to lose the game when simulating this.
        self.assertEqual(-1., value)
        self.assertFalse(node1.is_final)

        self.core.backpropagate(search_path, value)

        # Action of 8 yielded a reward of 0. The action has been discounted.
        # TODO: verify that the numbers are correct.
        np.testing.assert_almost_equal([1.2501018] * 8 + [-0.3749491],
                                       self.get_ucb_distribution(root))

        # We visited only action 8. The result is somewhat counter-intuitive so
        # far, but the policy is 100% on action 8.
        np.testing.assert_array_equal([0.] * 8 + [1.],
                                      self.core.get_policy_distribution())


if __name__ == '__main__':
    unittest.main()
