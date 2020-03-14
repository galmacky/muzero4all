
import unittest

from trajectory import Trajectory


class TrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.trajectory = Trajectory(discount=0.9)

    def test_make_target(self):
        self.trajectory.feed(0, 0., {0: -1., 1: 0.}, 0., [])
        self.trajectory.feed(1, 0., {0: 0., 1: -1.}, 0., [])
        self.assertEqual([(0.0, 0.0, {0: -1.0, 1: 0.0}), (0.0, 0.0, {0: 0.0, 1: -1.0})],
                         self.trajectory.make_target(state_index=0, num_unroll_steps=1, td_steps=2))


if __name__ == '__main__':
    unittest.main()
