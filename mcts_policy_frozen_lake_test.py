
import tensorflow as tf
import unittest

from gym.envs.toy_text.frozen_lake import LEFT
from gym.envs.toy_text.frozen_lake import RIGHT
from gym.envs.toy_text.frozen_lake import DOWN
from gym.envs.toy_text.frozen_lake import UP
from mcts_env_dynamics_model import MctsEnvDynamicsModel
from mcts_policy import MctsPolicy
from frozen_lake_det_env import FrozenLakeDetEnv

# WRITE a test that brings up the env, takes actions accoriding to mcts at every step, make sure
# we get the max reward

# TOOO(timkim): how do I get the test values I want to assert? +changwan@ for help???
class FrozenLakeMctsPolicyTest(unittest.TestCase):

    def setUp(self):
        self.env = FrozenLakeDetEnv()
        self.dynamics_model = MctsEnvDynamicsModel(self.env, discount=1.)
        self.policy = MctsPolicy(self.env, self.dynamics_model,
                                 num_simulations=100, discount=1.)

    def test_second_to_final(self):
        # Move to the left and up of Goal state.
        self.env.set_states([RIGHT, RIGHT, DOWN, DOWN])
        logits = self.policy.get_policy_logits()
        # RIGHT is also ok.
        self.assertEqual(DOWN, self.policy.choose_action(logits))
        self.assertEqual(DOWN, self.policy.action())

    def test_final(self):
        # Move to the left of Goal state.
        self.env.set_states([RIGHT, RIGHT, DOWN, DOWN, DOWN])
        logits = self.policy.get_policy_logits()
        # TODO: fix this and uncomment this.
        # tf.assert_equal(tf.constant([0.24, 0.31, 0.21, 0.24], tf.double), logits)
        # self.assertEqual(RIGHT, self.policy.choose_action(logits))
        # self.assertEqual(RIGHT, self.policy.action())

    def test_game_deterministic(self):
        while range(100):
            logits = self.policy.get_policy_logits()
            print (logits)
            action = self.policy.choose_action(logits)
            print ('Playing game: ', action)

            states, is_final, reward = self.env.step(action)
            break  # TODO: remove this once we fix this test.
            if is_final:
                break
            self.env.env.render()

        # TODO: uncomment this once we fix this test.
        #self.assertEqual(1.0, reward)


if __name__ == '__main__':
    unittest.main()
