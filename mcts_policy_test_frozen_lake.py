
import tensorflow as tf
import unittest

from mcts_env_dynamics_model import MctsEnvDynamicsModel
from mcts_policy import MctsPolicy
from frozen_lake_det_env import FrozenLakeEnv

# WRITE a test that brings up the env, takes actions accoriding to mcts at every step, make sure
# we get the max reward

# TOOO(timkim): how do I get the test values I want to assert? +changwan@ for help???
class FrozenLakeMctsPolicyTest(unittest.TestCase):
    def setUp(self):
        self.env = FrozenLakeEnv()
        print("@@@@@@@@@@@@@@@@@@@")
        self.dynamics_model = MctsEnvDynamicsModel(self.env)
        self.policy = MctsPolicy(self.env, self.dynamics_model, num_simulations=100)

    def test_game_deterministic(self):
        while True:
            #print (self.policy.get_policy_logits())
            action, states_isfinal_reward = self.policy.action()
            print ('Playing game: ', action, states_isfinal_reward)
            states, is_final, reward = states_isfinal_reward
            if is_final:
                break
        self.assertEqual(1.0, reward)

if __name__ == '__main__':
    unittest.main()
