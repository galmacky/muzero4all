
import unittest

from mcts_core import MctsCore
from tic_tac_toe_env import TicTacToeEnv


class MctsCoreTicTacToeTest(unittest.TestCase):

    def setUp(self):
        self.env = TicTacToeEnv()
        self.core = MctsCore(num_simulations=1, env=self.env)

    def test_game(self):
        self.core.initialize([0] * 9)
        # TODO: run tic-tac-toe game 100 times and check winning ratios.

    def test_rollout(self):
        self.core.initialize([0] * 9)
        # TODO: test more succinctly.
        self.assertEqual('{v: 0, p: 1.0, v_sum: 0, s: [0, 0, 0, 0, 0, 0, 0, 0, 0], r: 0, c: {0: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 1: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 2: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 3: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 4: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 5: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 6: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 7: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 8: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}}}',
                         str(self.core.get_root_for_testing()))
        self.core.rollout()
        # TODO: test more succinctly.
        self.assertEqual('{v: 1, p: 1.0, v_sum: 0.0, s: [0, 0, 0, 0, 0, 0, 0, 0, 0], r: 0, c: {0: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 1: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 2: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 3: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 4: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 5: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 6: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 7: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 8: {v: 1, p: 1.0, v_sum: 0.0, s: [4, 0, 0, 0, 0, 0, 0, 0, 1], r: 0.0, c: {0: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 1: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 2: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 3: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 4: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 5: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 6: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 7: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 8: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}}}}}',
                         str(self.core.get_root_for_testing()))
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
        self.core.initialize([0] * 9)
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
        self.assertEqual(0., value)
        self.assertFalse(node1.is_final)

        self.core.backpropagate(search_path, value)

        # Action of 8 yielded a reward of 0. The action has been discounted.
        # TODO: verify that the numbers are correct.
        self.assertEqual([1.25] * 8 + [0.625], self.get_ucb_distribution(root))


if __name__ == '__main__':
    unittest.main()
