
import time
import unittest

from basic_mcts_model import BasicMctsModel
from mcts_policy import MctsPolicy
from pacman_det_env import PacmanDetEnv
from gym.envs.atari import atari_env


class MctsPolicyFrozenLakeTest(unittest.TestCase):

    # RUN NOTES: (max_depth, num_simulations) and compute time.
    # 25, 25 worked pretty well, ~.5s per step
    # 10, 10 pretty fast, ~.1s per step, 1.0 discount achieves ~2200 score.
    def setUp(self):
        self.env = PacmanDetEnv()
        self.model = BasicMctsModel(self.env, discount=.999, max_depth=10)
        self.policy = MctsPolicy(self.env, self.model,
                                 num_simulations=25, discount=.999)

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
        idx = 0
        total_reward = 0
        while True:
            start_time = time.time()
            print('Starting action calculation')
            action = self.policy.action()
            end_time = time.time()
            states, is_final, reward = self.env.step(action)
            total_reward += reward
            print('Action at iter %s: %s\nReward: %s\n'
                'TotalReward: %s\nCalc time: %s\n\n' 
                % (idx, action, reward, total_reward, 
                    end_time - start_time))
            self.env.env.render()
            if is_final:
                print("Hit is_final!")
                break
            idx += 1
        # for 10, 10 got over 2200 with discount 1
        # with discount 9, seems to not die, but runs longer, and pacman
        # cuts off the game at iter 900
        self.assertTrue(total_reward > 2200)


if __name__ == '__main__':
    unittest.main()
