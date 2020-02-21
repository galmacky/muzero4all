

import gym

from frozen_lake_det_env import FrozenLakeEnv

fl_env = FrozenLakeEnv()
env = fl_env.env
for _ in range(10):
	env.render()
	# Take random action.
	fl_env.step(env.action_space.sample())

env.close()