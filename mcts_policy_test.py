
import tensorflow as tf
import unittest

from mcts_env_dynamics_model import MctsEnvDynamicsModel
from mcts_policy import MctsPolicy
from tic_tac_toe_env import TicTacToeEnv


class MctsPolicyTest(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()
        self.dynamics_model = MctsEnvDynamicsModel(self.env)
        self.policy = MctsPolicy(self.env, self.dynamics_model, num_simulations=100)

    def test_action(self):
        action, unused_env_step = self.policy.action()
        self.assertEqual(3, action)

    def test_policy_logits(self):
        logits = self.policy.get_policy_logits()
        tf.assert_equal(tf.constant([0.1, 0.1, 0.12, 0.12, 0.12, 0.12, 0.12, 0.1, 0.1], dtype=tf.float64), logits)

    def test_game(self):
        while True:
            print (self.env.get_states())
            action, unused_env_step = self.policy.action()
            print (action, unused_env_step)
            states, is_final, reward = unused_env_step
            if is_final:
                break
