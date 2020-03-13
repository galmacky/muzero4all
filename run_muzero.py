


import time
import unittest

from basic_mcts_model import BasicMctsModel
from mcts_policy import MctsPolicy
from pacman_det_env import PacmanDetEnv
from gym.envs.atari import atari_env
from network_initializer import TicTacToeInitializer, AtariInitializer
from network import Network
from muzero_collection_policy import MuZeroCollectionPolicy
from muzero_eval_policy import ReplayBuffer, MuZeroEvalPolicy

TRAIN_ITERATIONS = 5

PLAY_ITERATIONS = 20 
NUM_SELF_PLAYS = 20

NUM_TRAIN_STEPS = 20
NUM_UNROLL_STEPS = 20

tic_tac_toe_initializer = TicTacToeInitializer()

env = PacmanDetEnv()
network = Network(tic_tac_toe_initializer)
replay_buffer = ReplayBuffer()
col_policy = MuZeroCollectionPolicy(env, network, replay_buffer)
eval_policy = MuZeroEvalPolicy(env, network, replay_buffer)

for train_iter in range(TRAIN_ITERATIONS):
    print('STARTING TRAINING ITERATION')
    for play_iter in range(PLAY_ITERATIONS):
        print('STARTING PLAY ITERATION')
        start_time = time.time()
        col_policy.run_self_play(NUM_SELF_PLAYS)
        end_time = time.time()
        print('Self Play Iteration Runtime: %s' % end_time - start_time)
    eval_policy.train(NUM_TRAIN_STEPS, NUM_UNROLL_STEPS)


idx = 0
total_reward = 0
while True:
    start_time = time.time()
    print('Starting action calculation')
    action = self.policy.action()
    end_time = time.time()
    action = eval_policy.action()
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