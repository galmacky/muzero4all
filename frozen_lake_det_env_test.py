

import gym

from frozen_lake_det_env import FrozenLakeEnv
from gym.envs.toy_text import frozen_lake, discrete

fl_env = FrozenLakeEnv()
env = fl_env.env
for _ in range(10):
	env.render()
	# Take random action.
	fl_env.step(env.action_space.sample())

print("Set State: ")
fl_env.reset()
actions = [
	frozen_lake.RIGHT,
	frozen_lake.RIGHT,
	frozen_lake.DOWN,
]

fl_env.set_states(actions)
env.render()

env.close()