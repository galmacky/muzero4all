import numpy as np
import tensorflow as tf

from network import Action
from policy import Policy


class MuZeroEvalPolicy(Policy):
    """Eval Policy for MuZero. Used for training and getting the 
    real eval action to take."""

    def __init__(self, env, network, replay_buffer):
        self.env = env
        # As this implementation is single-threaded, no SharedStorage
        # is needed, instead we only keep track of a single network.


        # # Create a new network every time or only at the very beginning?
        # # For now, only create at the very beginning
        # self.network_initializer = network_initializer

        self.network = network
        self.replay_buffer = replay_buffer
        # TODO(timkim): FIND VALUES
        # AdamOptimizer
        # TODO: TUNE 
        self.lr = 3e-4
        self.weight_decay = 1e-4

        # self.model = MuZeroMctsModel(env, self.network)
        # # env is used only for the action space.
        # self.core = MctsCore(env, self.model, discount=discount)
        # self.num_simulations = num_simulations
        # self.rng = rng

    # def reset(self):
    #     self.model.reset()

    # def get_policy_logits(self):
    #     self.core.initialize()
    #     for _ in range(self.num_simulations):
    #         self.core.rollout()
    #     policy_logits = tf.convert_to_tensor(self.core.get_policy_distribution())
    #     # policy_logits = tf.expand_dims(policy_logits, 0)  # batch_size=1
    #     # print (policy_logits)
    #     return policy_logits

    # def choose_action(self, logits):
    #     # tf.random.set_seed(self.r_seed)
    #     # action = tf.random.categorical(logits=tf.math.log(logits), num_samples=1, seed=self.r_seed)
    #     # self.r_seed += 1
    #     # action = tf.squeeze(action)
    #     # action = np.random.choice(a=np.array(logits))
    #     # return action
    #     # TODO: break tie randomly.
    #     action = tf.math.argmax(logits)
    #     return action

    # IMPORTANT!!!!!!: num_unroll_steps needs to match the size of the rollouts in
    # MCTS +changwan@
    def train(self, num_steps, num_unroll_steps):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        for i in range(num_steps):
            print("Train Step:", i)
            batch = self.replay_buffer.sample_batch(
                num_unroll_steps, td_steps=2  #TODO: TUNE td_steps
                )
            print('BATCH: ', batch)
            self.update_weights(batch)

    def update_weights(self, batch):

        loss = 0
        value_loss_contrib_sum = 0
        reward_loss_contrib_sum = 0
        policy_loss_contrib_sum = 0
        weight_reg_loss_contrib_sum = 0

        with tf.GradientTape() as tape:
            for image, actions, targets in batch:
                # Reshape the states to be -1 x n dimension: -1 being the outer batch dimension.
                image = np.array(image).reshape(-1, len(image))
                # Initial step, from the real observation.
                value, reward, policy_logits, hidden_state = self.network.initial_inference(
                    image)
                predictions = [(1.0, value, reward, policy_logits)]

                # Recurrent steps, from action and previous hidden state.
                for action in actions:
                    value, reward, policy_logits, hidden_state = self.network.recurrent_inference(
                        hidden_state, Action(action))
                    predictions.append((1.0 / len(actions), value, reward, policy_logits))

                    hidden_state = self.scale_gradient(hidden_state, 0.5)


                for prediction, target in zip(predictions, targets):
                    gradient_scale, value, reward, policy_logits = prediction
                    print('\nPREDICTIONS:\n')
                    print('gradient_scale: ', gradient_scale)
                    print('value: ', value)
                    print('reward: ', reward)
                    print('policy_logits: ', policy_logits)
                    target_value, target_reward, target_policy = target
                    print('\nTARGETS:\n')
                    print('target_value: ', target_value)
                    print('target_reward: ', target_reward)
                    print('target_policy: ', target_policy)

                    # print ('prediction:', prediction)
                    # print ('target:', target)
                    # TODO: fix reward / target_reward to be float32.
                    value_loss_contrib = self.scalar_loss(value, target_value) 
                    print('value_loss_contrib: ', value_loss_contrib)
                    reward_loss_contrib = self.scalar_loss(reward, target_reward)
                    print('reward_loss_contrib: ', reward_loss_contrib)
                    policy_loss_contrib = tf.nn.softmax_cross_entropy_with_logits(
                                                logits=policy_logits, labels=target_policy)
                    print('policy_loss_contrib: ', policy_loss_contrib)
                    value_loss_contrib_sum += value_loss_contrib
                    reward_loss_contrib_sum += reward_loss_contrib
                    policy_loss_contrib_sum += policy_loss_contrib

                    l = (
                        value_loss_contrib +
                        reward_loss_contrib +
                        policy_loss_contrib)

                    total_loss_contrib = self.scale_gradient(l, gradient_scale)
                    print('total_loss_contrib: ', total_loss_contrib)
                    loss += total_loss_contrib
            for weights in self.network.get_weights():
                weight_reg_loss_contrib =  self.weight_decay * tf.nn.l2_loss(weights)
                weight_reg_loss_contrib_sum += weight_reg_loss_contrib
                loss += weight_reg_loss_contrib
        
        # self.optimizer.minimize(lambda: loss, var_list=self.network.get_weights())
        # print('NETWORK_WEIGHTS: ', self.network.get_weights())
        gradients = tape.gradient(loss, self.network.get_weights())
        self.optimizer.apply_gradients(zip(gradients, self.network.get_weights()))
        print('value_loss_contrib_sum: ', value_loss_contrib_sum)
        print('reward_loss_contrib_sum: ', reward_loss_contrib_sum)
        print('policy_loss_contrib_sum: ', policy_loss_contrib_sum)
        print('weight_reg_loss_contrib_sum: ', weight_reg_loss_contrib_sum)


        print('loss', loss)

    def scalar_loss(self, y_true, y_pred):
        return tf.square(y_true - y_pred)
        # TODO: check if this is correct
        return tf.keras.losses.MSE(y_true, y_pred)

    def scale_gradient(self, tensor, scale):
        """Scales the gradient for the backward pass."""
        return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)

    def get_policy_logits(self):
        current_state = self.env.get_current_game_input()
        current_state = np.expand_dims(current_state, 0)
        # policy_logits, value = self.network.prediction_network(
        #     self.network.initial_inference(current_state))
        network_output = self.network.initial_inference(current_state)
        return network_output.policy_logits
        # return policy_logits

    def action(self):
        #arg max along logits dimension, not batch dimension, reshape to scalar tensor.
        return tf.reshape(tf.math.argmax(self.get_policy_logits(),axis=1), [])
