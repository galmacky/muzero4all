
import numpy as np
import tensorflow as tf

from mcts_core import MctsCore
from policy import Policy


class MctsPolicy(Policy):

    def __init__(self, env, dynamics_model, num_simulations=100, r_seed=0):
        self.core = MctsCore(env, dynamics_model)
        self.env = env
        self.dynamics_model = dynamics_model
        self.num_simulations = num_simulations
        self.r_seed_init = r_seed
        self.r_seed = r_seed

    def reset(self):
        self.dynamics_model.reset()
        # We may need to reset env as well.
        self.r_seed = self.r_seed_init

    def get_policy_logits(self):
        self.core.initialize()
        for _ in range(self.num_simulations):
            self.core.rollout()
        policy_logits = tf.convert_to_tensor(self.core.get_policy_distribution())
        #policy_logits = tf.expand_dims(policy_logits, 0)  # batch_size=1
        return policy_logits

    def sample_action(self, logits):
        #tf.random.set_seed(self.r_seed)
        #action = tf.random.categorical(logits=tf.math.log(logits), num_samples=1, seed=self.r_seed)
        #self.r_seed += 1
        #action = tf.squeeze(action)
        #action = np.random.choice(a=np.array(logits))
        #return action
        # TODO: break tie randomly.
        action = tf.math.argmax(logits)
        return action

    def action(self):
        sampled_action = self.sample_action(self.get_policy_logits())
        return sampled_action, self.env.step(sampled_action)
