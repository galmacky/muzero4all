
import unittest

from gym.envs.toy_text.frozen_lake import LEFT
from gym.envs.toy_text.frozen_lake import RIGHT
from gym.envs.toy_text.frozen_lake import DOWN
from gym.envs.toy_text.frozen_lake import UP
from mcts_env_dynamics_model import MctsEnvDynamicsModel
from mcts_policy import MctsPolicy
from frozen_lake_det_env import FrozenLakeDetEnv


class MctsPolicyFrozenLakeTest(unittest.TestCase):

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
        self.assertEqual(RIGHT, self.policy.action())

    def test_game_deterministic(self):
        while range(100):
            action = self.policy.action()
            states, is_final, reward = self.env.step(action)
            if is_final:
                break
        self.assertEqual(1.0, reward)


if __name__ == '__main__':
    unittest.main()
