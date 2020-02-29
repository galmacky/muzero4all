
import tensorflow as tf
import numpy as np
import unittest

from gym.envs.toy_text.frozen_lake import LEFT
from gym.envs.toy_text.frozen_lake import RIGHT
from gym.envs.toy_text.frozen_lake import DOWN
from gym.envs.toy_text.frozen_lake import UP
from mcts_core import MctsCore
from mcts_env_dynamics_model import MctsEnvDynamicsModel
from frozen_lake_det_env import FrozenLakeDetEnv


class MctsCoreFrozenLakeTest(unittest.TestCase):

    def setUp(self):
        self.env = FrozenLakeDetEnv()
        self.dynamics_model = MctsEnvDynamicsModel(self.env)
        self.core = MctsCore(env=self.env, dynamics_model=self.dynamics_model)

    def test_initial_rollout(self):
        self.core.initialize()
        # TODO: test more succinctly.
        self.assertEqual('{v: 0, p: 1.0, v_sum: 0, s: [], r: 0, c: {0: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 1: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 2: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 3: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}}}',
                         str(self.core.get_root_for_testing()))
        self.core.rollout()
        # TODO: test more succinctly.
        self.assertEqual('{v: 1, p: 1.0, v_sum: 0.0, s: [], r: 0, c: {0: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 1: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 2: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 3: {v: 1, p: 1.0, v_sum: 0.0, s: [3], r: 0.0, c: {0: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 1: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 2: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}, 3: {v: 0, p: 1.0, v_sum: 0, s: None, r: 0, c: {}}}}}}',
                         str(self.core.get_root_for_testing()))
        # rollout should not affect the actual test.
        self.assertEqual([], self.env.get_states())
        self.core.rollout()
        # TODO: check nodes after rollout

    def test_final_rollout(self):
        self.dynamics_model = MctsEnvDynamicsModel(self.env, discount=0.8)
        self.core = MctsCore(env=self.env, dynamics_model=self.dynamics_model, discount=0.8)

        self.env.set_states([RIGHT, RIGHT, DOWN, DOWN, DOWN])
        self.core.initialize()
        for _ in range(3000):
            self.core.rollout()
            # print (self.get_ucb_distribution(self.core.get_root_for_testing()))
        dist = self.get_ucb_distribution(self.core.get_root_for_testing())
        # print (self.get_ucb_distribution(self.core.get_root_for_testing()))
        # print (self.core.get_policy_distribution())
        # TODO: fix and uncomment this.
        # self.assertEqual(RIGHT, tf.argmax(dist))

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

        self.assertEqual([0.] * 4, self.get_ucb_distribution(root))

        node1, search_path, last_action = self.core.select_node()

        self.assertNotEqual(root, node1)
        self.assertEqual([root, node1], search_path)
        # We can choose any action since the ucb distribution is uniform over actions.
        self.assertEqual(3, last_action)

        self.core.expand_node(node1)

        self.assertTrue(node1.expanded())
        parent = root
        self.assertIsNotNone(parent.states)

        value = self.core.evaluate_node(node1, parent.states, last_action)

        self.assertEqual(0., node1.reward)
        self.assertEqual([3], node1.states)
        # It is likely to lose the game when simulating this.
        self.assertEqual(0., value)
        self.assertFalse(node1.is_final)

        self.core.backpropagate(search_path, value)

        # Action of 8 yielded a reward of 0. The action has been discounted.
        # TODO: verify that the numbers are correct.
        np.testing.assert_almost_equal([1.2501018, 1.2501018, 1.2501018, 0.6250509],
                                       self.get_ucb_distribution(root))

        # We visited only action 3. The result is somewhat counter-intuitive so
        # far, but the policy is 100% on action 8.
        np.testing.assert_array_equal([0.] * 3 + [1.],
                                      self.core.get_policy_distribution())

    def test_inside_final_rollouts(self):
        self.env.set_states([RIGHT, RIGHT, DOWN, DOWN, DOWN])
        self.core.initialize()
        root = self.core.get_root_for_testing()

        self.assertEqual([0.] * 4, self.get_ucb_distribution(root))

        node1, search_path, last_action = self.core.select_node()

        self.assertNotEqual(root, node1)
        self.assertEqual([root, node1], search_path)
        # We can choose any action since the ucb distribution is uniform over actions.
        self.assertEqual(3, last_action)

        self.core.expand_node(node1)

        self.assertTrue(node1.expanded())
        parent = root
        self.assertIsNotNone(parent.states)

        value = self.core.evaluate_node(node1, parent.states, last_action)

        self.assertEqual(0., node1.reward)
        self.assertEqual([RIGHT, RIGHT, DOWN, DOWN, DOWN, UP], node1.states)
        # It is likely to lose the game when simulating this.
        self.assertEqual(0., value)
        self.assertFalse(node1.is_final)

        self.core.backpropagate(search_path, value)

        # Action of 8 yielded a reward of 0. The action has been discounted.
        # TODO: verify that the numbers are correct.
        np.testing.assert_almost_equal([1.2501018, 1.2501018, 1.2501018, 0.6250509],
                                       self.get_ucb_distribution(root))

        # We visited only action 3. The result is somewhat counter-intuitive so
        # far, but the policy is 100% on action 8.
        np.testing.assert_array_equal([0.] * 3 + [1.],
                                      self.core.get_policy_distribution())


        node1, search_path, last_action = self.core.select_node()

        self.assertNotEqual(root, node1)
        self.assertEqual([root, node1], search_path)
        # We can choose any action since the ucb distribution is uniform over actions.
        self.assertEqual(2, last_action)

        self.core.expand_node(node1)

        self.assertTrue(node1.expanded())
        parent = root
        self.assertIsNotNone(parent.states)

        value = self.core.evaluate_node(node1, parent.states, last_action)

        self.assertEqual(1., node1.reward)
        self.assertEqual([RIGHT, RIGHT, DOWN, DOWN, DOWN, RIGHT], node1.states)
        # It is likely to lose the game when simulating this.
        self.assertEqual(1., value)
        self.assertTrue(node1.is_final)

        self.core.backpropagate(search_path, value)

        # Action of 3 yielded a reward of 0. The action has been discounted.
        # TODO: verify that the numbers are correct.
        np.testing.assert_almost_equal([1.7679828, 1.7679828, 1.8839914, 0.8839914],
                                       self.get_ucb_distribution(root))

        # We visited only action 2. The result is somewhat counter-intuitive so
        # far, but the policy is 100% on action 8.
        np.testing.assert_array_equal([0., 0., 0.5, 0.5],
                                      self.core.get_policy_distribution())


if __name__ == '__main__':
    unittest.main()
