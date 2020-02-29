
import re
import unittest

from frozen_lake_det_env import FrozenLakeEnv
from gym.envs.toy_text import frozen_lake


class TicTacToeMctsPolicyTest(unittest.TestCase):

    def setUp(self):
        self.env = FrozenLakeEnv()

    def tearDown(self):
        self.env.env.close()

    def get_real_states(self):
        """Returns env's real states not the action history."""
        return self.remove_ansi(self.env.env.render(mode='ansi').getvalue())
        #    return rendered_textiowrapper.read()

    def remove_ansi(self, ansi_str):
        ansi_escape_8bit = re.compile(
            r'(?:\x1B[@-Z\\-_]|[\x80-\x9A\x9C-\x9F]|(?:\x1B\[|\x9B)[0-?]*[ -/]*[@-~])'
        )
        return ansi_escape_8bit.sub('', ansi_str)


    def test_init(self):
        self.assertEqual('\nSFFF\nFHFH\nFFFH\nHFFG\n', self.get_real_states())

    def test_win(self):
        actions = [
            frozen_lake.RIGHT,
            frozen_lake.RIGHT,
            frozen_lake.DOWN,
            frozen_lake.DOWN,
            frozen_lake.DOWN,
        ]

        for action in actions:
            new_states, is_final, reward = self.env.step(action)

        self.assertFalse(is_final)
        self.assertEqual(0.0, reward)
        self.assertEqual(actions, new_states)
        self.assertEqual('  (Down)\nSFFF\nFHFH\nFFFH\nHFFG\n', self.get_real_states())

        new_states, is_final, reward = self.env.step(frozen_lake.RIGHT)

        self.assertTrue(is_final)
        self.assertEqual(1.0, reward)
        self.assertEqual(actions + [frozen_lake.RIGHT], new_states)
        self.assertEqual('  (Right)\nSFFF\nFHFH\nFFFH\nHFFG\n', self.get_real_states())

    def test_hole(self):
        actions = [
            frozen_lake.RIGHT,
            frozen_lake.DOWN,
        ]

        for action in actions:
            new_states, is_final, reward = self.env.step(action)
        self.assertTrue(is_final)
        self.assertEqual(0.0, reward)
        self.assertEqual(actions, new_states)
        self.assertEqual('  (Down)\nSFFF\nFHFH\nFFFH\nHFFG\n', self.get_real_states())


if __name__ == '__main__':
    unittest.main()
