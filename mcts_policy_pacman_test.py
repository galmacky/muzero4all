
import time
import unittest

from basic_mcts_model import BasicMctsModel
from mcts_policy import MctsPolicy
from pacman_det_env import PacmanDetEnv
from gym.envs.atari import atari_env


class MctsPolicyFrozenLakeTest(unittest.TestCase):

    def setUp(self):
        self.env = PacmanDetEnv()
        self.model = BasicMctsModel(self.env, discount=1., max_depth=10)
        self.policy = MctsPolicy(self.env, self.model,
                                 num_simulations=10, discount=1.)

    # def test_second_to_final(self):
    #     # Move to the left and up of Goal state.
    #     self.env.set_states([RIGHT, RIGHT, DOWN, DOWN])
    #     logits = self.policy.get_policy_logits()
    #     # RIGHT is also ok.
    #     self.assertEqual(DOWN, self.policy.choose_action(logits))
    #     self.assertEqual(DOWN, self.policy.action())

    # def test_final(self):
    #     # Move to the left of Goal state.
    #     self.env.set_states([RIGHT, RIGHT, DOWN, DOWN, DOWN])
    #     self.assertEqual(RIGHT, self.policy.action())

    def test_game_deterministic(self):
        for idx in range(100):
            start_time = time.time()
            print('Starting action calculation')
            action = self.policy.action()
            end_time = time.time()
            states, is_final, reward = self.env.step(action)
            print('Action at iter %s: %s\nReward: %s\nCalc time: %s' 
                % (idx, action, reward, end_time - start_time))
            if is_final:
                break
            self.env.env.render()
        self.assertEqual(1.0, reward)


if __name__ == '__main__':
    unittest.main()
