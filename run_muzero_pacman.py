
import time
import unittest
import tensorflow as tf

from basic_mcts_model import BasicMctsModel
from mcts_policy import MctsPolicy
from pacman_det_env import PacmanDetEnv
from gym.envs.atari import atari_env
from network_initializer import TicTacToeInitializer, AtariInitializer, PacManInitializer
from network import Network
from muzero_collection_policy import MuZeroCollectionPolicy
from muzero_eval_policy import MuZeroEvalPolicy
from replay_buffer import ReplayBuffer
from tic_tac_toe_env import TicTacToeEnv


TRAIN_ITERATIONS = 200

PLAY_ITERATIONS = 1

NUM_TRAIN_STEPS = 20
NUM_UNROLL_STEPS = 5

initializer = PacManInitializer()
env = PacmanDetEnv(screen_images=True)
env.reset()

#env = PacmanDetEnv()
network = Network(initializer)
replay_buffer = ReplayBuffer()
col_policy = MuZeroCollectionPolicy(env, network, replay_buffer,
                                    num_simulations=100, max_moves=2000, discount=0.999)
eval_policy = MuZeroEvalPolicy(env, network, replay_buffer)

eval_log_dir = 'logs/gradient_tape/eval'
eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

for train_iter in range(TRAIN_ITERATIONS):
    print('STARTING TRAINING ITERATION #{}'.format(train_iter))
    tf.summary.experimental.set_step(train_iter)
    for play_iter in range(PLAY_ITERATIONS):
        print('STARTING PLAY ITERATION #{}'.format(play_iter))
        start_time = time.time()
        col_policy.run_self_play(render=True)
        end_time = time.time()
        print('Self Play Iteration Runtime: {}'.format(end_time - start_time))
    eval_policy.train(NUM_TRAIN_STEPS, NUM_UNROLL_STEPS)
    # TODO: pacman, save weights, tensorboard

    total_reward = 0
    env.reset()
    for idx in range(100000):
        start_time = time.time()
        print('Starting action calculation')
        action = eval_policy.action()
        states, is_final, reward = env.step(action)
        total_reward += reward
        end_time = time.time()
        print('Action at iter %s: %s\nReward: %s\n'
            'TotalReward: %s\nCalc time: %s\n\n'
            % (idx, action, reward, total_reward,
                end_time - start_time))
        env.render()
        if is_final:
            print("Hit is_final!")
            break
        idx += 1
    last_trajectory = replay_buffer.buffer[-1]
    total_reward_2 = sum(last_trajectory.rewards)
    with eval_summary_writer.as_default():
        tf.summary.scalar('total_reward', total_reward, step=train_iter)
        tf.summary.scalar('training_total_reward', total_reward_2, step=train_iter)

