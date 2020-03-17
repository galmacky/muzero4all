
import unittest

from trajectory import Trajectory


class TrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.trajectory = Trajectory(discount=0.9)

    def test_make_target(self):
        self.trajectory.feed(action=0, reward=0., child_visit_dist={0: 0.1, 1: 0.9}, root_value=0., game_state=[])
        self.trajectory.feed(action=1, reward=1., child_visit_dist={0: 0.2, 1: 0.8}, root_value=0.9, game_state=[])
        # (value, last_reward, child_visit_dist)
        self.assertEqual([(0.9, 0.0, {0: 0.1, 1: 0.9}), (1.0, 0.0, {0: 0.2, 1: 0.8})],
                         self.trajectory.make_target(state_index=0, num_unroll_steps=1, td_steps=2))


if __name__ == '__main__':
    unittest.main()
