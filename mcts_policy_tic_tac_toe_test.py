
import collections
import tensorflow as tf
import unittest

from mcts_env_dynamics_model import MctsEnvDynamicsModel
from mcts_policy import MctsPolicy
from tic_tac_toe_env import TicTacToeEnv


class MctsPolicyTicTacToeTest(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()
        self.dynamics_model = MctsEnvDynamicsModel(self.env)
        self.policy = MctsPolicy(self.env, self.dynamics_model, num_simulations=100)

    def test_action_start(self):
        action = self.policy.action()
        states_isfinal_reward = self.env.step(action)
        self.assertEqual(0, action)
        self.assertEqual(([1, 4, 0, 0, 0, 0, 0, 0, 0], False, 0.0), states_isfinal_reward)

    def test_action_win(self):
        self.env.set_states([1, 0, 1,
                             1, 0, 4,
                             4, 4, 0])
        action = self.policy.action()
        states_isfinal_reward = self.env.step(action)
        self.assertEqual(1, action)
        self.assertEqual(([1, 1, 1,
                           1, 0, 4,
                           4, 4, 0], True, 1.0), states_isfinal_reward)

    def test_action_win_2(self):
        self.env.set_states([1, 1, 4,
                             0, 0, 4,
                             1, 4, 0])
        action = self.policy.action()
        states_isfinal_reward = self.env.step(action)
        self.assertEqual(3, action)
        self.assertEqual(([1, 1, 4,
                           1, 0, 4,
                           1, 4, 0], True, 1.0), states_isfinal_reward)

    def test_policy_logits(self):
        logits = self.policy.get_policy_logits()
        tf.assert_equal(tf.constant([0.14, 0.09, 0.13, 0.09, 0.13, 0.11, 0.09, 0.11, 0.11],
                                    dtype=tf.float64), logits)

    def test_choose_action(self):
        self.assertEqual(1, self.policy.choose_action(tf.constant(
            [0.11, 0.116, 0.11, 0.11, 0.11, 0.111, 0.111, 0.111, 0.111])))

    def test_game_deterministic(self):
        while True:
            action = self.policy.action()
            states_isfinal_reward = self.env.step(action)
            states, is_final, reward = states_isfinal_reward
            if is_final:
                break
        self.assertEqual(1.0, reward)

    def play_game_once(self, r_seed):
        self.env = TicTacToeEnv(use_random=True, r_seed=r_seed)
        self.dynamics_model = MctsEnvDynamicsModel(self.env, r_seed=r_seed)
        self.policy = MctsPolicy(self.env, self.dynamics_model, num_simulations=100,
                                 r_seed=r_seed)
        while True:
            action = self.policy.action()
            states_isfinal_reward = self.env.step(action)
            states, is_final, reward = states_isfinal_reward
            if is_final:
                return states, is_final, reward

    def test_game_random(self):
        reward_dict = collections.defaultdict(int)
        for r_seed in range(100):
            _, _, reward = self.play_game_once(r_seed)
            reward_dict[reward] += 1
        print ('reward distribution: ', reward_dict)
        # 96% winning ratio.
        self.assertEqual({1.0: 96, 0.0: 1, -1.0: 3}, reward_dict)


if __name__ == '__main__':
    unittest.main()
