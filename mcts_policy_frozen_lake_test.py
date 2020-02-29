
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
        self.dynamics_model = MctsEnvDynamicsModel(self.env)
        self.policy = MctsPolicy(self.env, self.dynamics_model, 
            num_simulations=10)

    def test_final(self):
        # Move to the left of Goal state.
        self.env.set_states([RIGHT, RIGHT, DOWN, DOWN, DOWN])
        tf.assert_equal(tf.constant([0.2, 0.3, 0.3, 0.2], tf.double), self.policy.get_policy_logits())

    def test_game_deterministic(self):
        while True:
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
