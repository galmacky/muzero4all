import tensorflow as tf

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
        self.lr = 3e-2
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
            batch = self.replay_buffer.sample_batch(
                num_unroll_steps, td_steps=10  #TODO: TUNE td_steps
                )
            self.update_weights(self.optimizer, batch)

    def update_weights(self, batch):

        loss = 0
        for image, actions, targets in batch:
            # Initial step, from the real observation.
            value, reward, policy_logits, hidden_state = self.network.initial_inference(
                image)
            predictions = [(1.0, value, reward, policy_logits)]

            # Recurrent steps, from action and previous hidden state.
            for action in actions:
                value, reward, policy_logits, hidden_state = network.recurrent_inference(
                    hidden_state, action)
                predictions.append((1.0 / len(actions), value, reward, policy_logits))

                hidden_state = tf.scale_gradient(hidden_state, 0.5)

            for prediction, target in zip(predictions, targets):
                gradient_scale, value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target

                l = (
                    scalar_loss(value, target_value) +
                    scalar_loss(reward, target_reward) +
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=policy_logits, labels=target_policy))

                loss += tf.scale_gradient(l, gradient_scale)

        for weights in network.get_weights():
            loss += self.weight_decay * tf.nn.l2_loss(weights)
  
        optimizer.minimize(loss)

    def get_policy_logits(self):
        current_state = self.env.get_current_game_input()
        policy_logits, value = self.network.prediction_network(
            self.network.initial_inference(current_state))
        return policy_logits

    def action(self):
        return tf.math.argmax(self.get_policy_logits())
