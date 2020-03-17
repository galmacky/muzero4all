
import numpy as np
import tensorflow as tf

from mcts_core import MctsCore
from muzero_mcts_model import MuZeroMctsModel
from policy import Policy
from network import Network
from trajectory import Trajectory


class MuZeroCollectionPolicy(Policy):
    """Policy for MuZero."""

    def __init__(self, env, network, replay_buffer, max_moves=1000,
                 num_simulations=100, discount=1.,
                 rng: np.random.RandomState = np.random.RandomState()):
        self.network = network
        self.replay_buffer = replay_buffer
        self.max_moves = max_moves
        self.env = env
        self.model = MuZeroMctsModel(env, self.network)  # TODO: investigate these values?
        self.discount = discount
        # env is used only for the action space.
        self.core = MctsCore(env, self.model, discount=discount)
        self.num_simulations = num_simulations
        self.rng = rng

    def reset(self):
        self.model.reset()

    def get_policy_logits(self): # ??WRONG?
        self.core.initialize()
        self.core.add_exploration_noise()
        for _ in range(self.num_simulations):
            self.core.rollout()
        policy_logits = tf.convert_to_tensor(self.core.get_policy_distribution())
        # policy_logits = tf.expand_dims(policy_logits, 0)  # batch_size=1
        # print (policy_logits)
        return policy_logits

    def choose_action(self, logits):
        # tf.random.set_seed(self.r_seed)
        # action = tf.random.categorical(logits=tf.math.log(logits), num_samples=1, seed=self.r_seed)
        # self.r_seed += 1
        # action = tf.squeeze(action)
        # action = np.random.choice(a=np.array(logits))
        # return action
        # TODO: break tie randomly.
        action = tf.math.argmax(logits)
        return action

    def action(self):
        return self.choose_action(self.get_policy_logits())

    def run_self_play(self, render=False):
        self.env.reset()
        trajectory = Trajectory(discount=self.discount)
        print("####### START run_self_play")
        for _ in range(self.max_moves):
            p = self.get_policy_logits()
            v = self.core.get_value()
            best_action = self.choose_action(p)
            observation = self.env.get_current_game_input()  # Use game state before taking the action.
            states, is_final, reward = self.env.step(best_action)
            if render:
                self.env.render()
            ######## BUG HERE!!!!!!!!!!! ISFINAL NEVER TRIGGERED
            # print('self play env step res (states, is_final, reward): ',
            #  states, is_final, reward)
            trajectory.feed(best_action, reward, p, v, observation)
            if is_final:
                print("#######")
                break
        # print('!!!!!!!!!!!!!!!!!!!!!!!\n')
        # print('TRAJECTORY ACTIONS: ', trajectory.action_history)
        # print('TRAJECTORY CHILD_VISITS: ', trajectory.child_visits)
        # print('TRAJECTORY REWARDS: ', trajectory.rewards)
        self.feed_replay_buffer(trajectory)

    def feed_replay_buffer(self, trajectory):
        self.replay_buffer.save_game(trajectory)
