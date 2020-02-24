
import tensorflow as tf
import unittest

from mcts_env_dynamics_model import MctsEnvDynamicsModel
from mcts_policy import MctsPolicy
from frozen_lake_det_env import FrozenLakeEnv

# TOOO(timkim): how do I get the test values I want to assert? +changwan@ for help???
class FrozenLakeMctsPolicyTest(unittest.TestCase):
    def setUp(self):
        self.env = FrozenLakeEnv()
        self.dynamics_model = MctsEnvDynamicsModel(self.env)
        self.policy = MctsPolicy(self.env, self.dynamics_model, num_simulations=100)

    def test_action_start(self):
        action, states_isfinal_reward = self.policy.action()
        self.assertEqual(0, action)
        self.assertEqual(([1, 4, 0, 0, 0, 0, 0, 0, 0], False, 0.0), states_isfinal_reward)

    def test_action_win(self):
        self.env.set_states([1, 0, 1,
                             1, 0, 4,
                             4, 4, 0])
        action, states_isfinal_reward = self.policy.action()
        self.assertEqual(1, action)
        self.assertEqual(([1, 1, 1,
                           1, 0, 4,
                           4, 4, 0], True, 1.0), states_isfinal_reward)

    def test_action_win_2(self):
        self.env.set_states([1, 1, 4,
                             0, 0, 4,
                             1, 4, 0])
        action, states_isfinal_reward = self.policy.action()
        self.assertEqual(3, action)
        self.assertEqual(([1, 1, 4,
                           1, 0, 4,
                           1, 4, 0], True, 1.0), states_isfinal_reward)

    def test_policy_logits(self):
        logits = self.policy.get_policy_logits()
        tf.assert_equal(tf.constant([0.13, 0.09, 0.13, 0.09, 0.13, 0.11, 0.1, 0.11, 0.11],
                                    dtype=tf.float64), logits)

    def test_sample_action(self):
        self.assertEqual(1, self.policy.sample_action(tf.constant(
            [0.11, 0.116, 0.11, 0.11, 0.11, 0.111, 0.111, 0.111, 0.111])))

    def test_game_deterministic(self):
        while True:
            #print (self.policy.get_policy_logits())
            action, states_isfinal_reward = self.policy.action()
            print ('Playing game: ', action, states_isfinal_reward)
            states, is_final, reward = states_isfinal_reward
            if is_final:
                break
        self.assertEqual(1.0, reward)

    def test_game_random(self):
        # TODO: this fails when r_seed=7. Fix this.
        for r_seed in range(5):
            self.env = TicTacToeEnv(use_random=True, r_seed=r_seed)
            self.dynamics_model = MctsEnvDynamicsModel(self.env, r_seed=r_seed)
            self.policy = MctsPolicy(self.env, self.dynamics_model, num_simulations=100,
                                     r_seed=r_seed)
            while True:
                action, states_isfinal_reward = self.policy.action()
                print ('Playing game: ', action, states_isfinal_reward)
                states, is_final, reward = states_isfinal_reward
                if is_final:
                    break
            self.assertEqual(1.0, reward, 'Failed with r_seed: {}'.format(r_seed))
